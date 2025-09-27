from llm_sdk import LightweightCausalLM
from .inputs import (
    FunctionDefinition,
    load_function_definitions,
    load_prompts,
)
import json
from pathlib import Path
import regex as re


def _bytes_to_unicode_mapping() -> dict[int, str]:
    """GPT-2 と同等の可逆バイト→Unicodeマップを生成する。"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return {b: c for b, c in zip(bs, cs)}


def _build_byte_encoder_decoder() -> tuple[dict[int, str], dict[str, int]]:
    byte_encoder = _bytes_to_unicode_mapping()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder


def load_merges_to_ranks(merges_path: str) -> dict[tuple[str, str], int]:
    """merges.txt を BPE の pair→rank に変換する。

    各行は "A B" の2トークン（byte-level unicode 記号列）。
    """
    lines = [
        ln.strip() for ln in Path(merges_path).read_text(encoding="utf-8").splitlines()
    ]
    pairs = [ln for ln in lines if ln and not ln.startswith("#")]  # 空行・コメント除外
    bpe_ranks: dict[tuple[str, str], int] = {}
    for i, ln in enumerate(pairs):
        a, b = ln.split()
        bpe_ranks[(a, b)] = i
    return bpe_ranks


def encode(
    text: str,
    bpe_ranks: dict[tuple[str, str], int],
    token_to_id: dict[str, int],
    byte_encoder: dict[int, str],
) -> list[int]:
    def get_pairs(symbols: list[str]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        prev = symbols[0]
        for ch in symbols[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def bpe(token: str) -> list[str]:
        if not token:
            return []
        symbols = list(token)
        # BPEなので隣接ペアを取得
        pairs = get_pairs(symbols)
        if not pairs:
            return [token]
        while True:
            min_rank = None
            best_pair = None
            # rankが最小のペアを取得
            # rank=学習時のそのステップで最頻だったペア
            for pair in pairs:
                rank = bpe_ranks.get(pair)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            first, second = best_pair
            new_symbols: list[str] = []
            i = 0
            # 最頻ペアを合わせて新しいsymbolsを作成し、圧縮していく
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(first + second)
                    i += 2
                    continue
                new_symbols.append(symbols[i])
                i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break
            pairs = get_pairs(symbols)
        return symbols

    # Qwen3の正規表現でプレトークナイズ
    qwen3_pattern = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    words = qwen3_pattern.findall(text)
    ids: list[int] = []
    for word in words:
        # 様々な文字列に対して、UTF-8にエンコードした後、byte_encoderでUnicodeに変換
        # なぜbyte_encoderでUnicodeに変換するかというと、merges.txtにはUTF-8のバイト列で書かれているため
        encoded = "".join(byte_encoder[b] for b in word.encode("utf-8"))
        # BPEで圧縮し、qwen3の語彙(str → id)に変換
        for piece in bpe(encoded):
            ids.append(token_to_id[piece])
    return ids


def decode(
    tokens: list[int], id_to_token: dict[int, str], byte_decoder: dict[str, int]
) -> str:
    text = "".join(id_to_token[t] for t in tokens)
    byte_values = [byte_decoder[ch] for ch in text]
    return bytes(byte_values).decode("utf-8", errors="replace")


def main() -> None:
    # 入力、出力データのパスを設定
    base_path = Path(__file__).parent.parent
    functions_path = base_path / "input" / "functions_definition.json"
    prompts_path = base_path / "input" / "function_calling_tests.json"
    results_dir = base_path / "output"
    results_dir.mkdir(exist_ok=True)

    # functionの定義、プロンプトを読み込む
    function_definitions = load_function_definitions(functions_path)
    prompts = load_prompts(prompts_path)

    # BPEのpair→rankを読み込む
    bpe_ranks = load_merges_to_ranks(base_path / "input" / "merges.txt")

    # byte→Unicode写像（可逆）を作成
    byte_encoder, byte_decoder = _build_byte_encoder_decoder()

    # 関数定義を文字列に整形
    def format_fn(defn: FunctionDefinition) -> str:
        args_spec = ", ".join(
            f"{name}: {defn.args_types.get(name, 'any')}" for name in defn.args_names
        )
        return f"{defn.fn_name}({args_spec}) -> {defn.return_type}"

    # 関数定義を文字列に整形
    functions_catalog = "\n".join(format_fn(d) for d in function_definitions)

    # モデルを初期化
    model = LightweightCausalLM(
        model_id="Qwen/Qwen3-0.6B",
        device="cpu",
        dtype=None,
    )

    # vocab.json（token → id）
    vocab_path = model.get_path_to_vocabulary_json()
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: dict[str, int] = json.load(f)
    id_to_token: dict[int, str] = {v: k for k, v in token_to_id.items()}

    # 各プロンプトを処理して関数呼び出し用のJSONを生成
    results: list[dict] = []
    max_new_tokens = 256

    for item in prompts:
        print(f"\n処理中のプロンプト: {item.prompt}")

        # プロンプトを構築（関数一覧 + 質問 + JSON出力指示）
        # ロールプレイプロンプティング
        user_prompt = (
            "role: user \n"
            "You are a function selector. Read the available functions and the question, then respond with exactly ONE JSON object.\n"
            "- The output MUST be valid JSON with exactly two top-level keys: fn_name (string) and args (object).\n"
            "- args must ALWAYS be a JSON object (even for a single argument). Use key-value pairs with the exact argument names.\n"
            "Available functions:\n"
            + functions_catalog
            + "\n\n"
            + "Question:\n"
            + item.prompt
            + "\n\n"
        )
        assistant_prompt = "role: assistant \n" + "json content: "
        prompt_text = user_prompt + assistant_prompt

        print("生成されたプロンプト:")
        print(prompt_text)

        # 内部BPEでエンコード
        encoded = encode(prompt_text, bpe_ranks, token_to_id, byte_encoder)
        print("encoded", encoded)
        input_ids = encoded
        start_len = len(input_ids)

        print("\nトークン生成中...")

        json_start_flag = False
        resource = 0
        next_start = start_len

        # 貪欲法による生成
        for i in range(max_new_tokens):
            logits = model.get_logits_from_input_ids(input_ids)
            # argmax を取得（numpyは未使用）
            next_id = max(range(len(logits)), key=lambda i: logits[i])
            input_ids.append(int(next_id))
            # 単純な終了条件: '}' が出現
            # テキストを生成して逐次検索。json_start_flag が真かつ resource({}) が 0 になったら停止
            # TODO: O(n) なので手法を改善する
            generated_text = decode(input_ids[next_start:], id_to_token, byte_decoder)
            next_start += 1
            print("generated_text", generated_text)
            for char in generated_text:
                if char == "{":
                    json_start_flag = True
                    resource += 1
                elif char == "}":
                    resource -= 1
            if json_start_flag and resource == 0:
                print(f"生成完了 ({i + 1} トークン)")
                break

        # デコードしてJSONを抽出
        generated_text = decode(input_ids[start_len:], id_to_token, byte_decoder)
        # generated_text = model._decode(input_ids[start_len:])
        print("\n生成されたテキスト:")
        print(generated_text)

        # JSONをパース
        json_obj: dict
        try:
            start = generated_text.find("{")
            end = generated_text.rfind("}")
            payload = (
                generated_text[start : end + 1]
                if start != -1 and end != -1 and end > start
                else "{}"
            )
            json_obj = json.loads(payload)
            print("\n抽出されたJSON:")
            print(json.dumps(json_obj, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"\nJSONパース失敗: {e}")
            json_obj = {"fn_name": "", "args": {}}

        # args が配列になる場合があるためここで検証
        fn_name = json_obj.get("fn_name", "")
        args = json_obj.get("args", {})
        if not isinstance(args, dict):
            args = {}

        results.append(
            {
                "prompt": item.prompt,
                "fn_name": fn_name,
                "args": args,
            }
        )

    print("\n全プロンプトの処理が完了しました")

    # 出力を保存
    output_path = results_dir / "function_calling_name.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {output_path}")

    return


if __name__ == "__main__":
    main()
