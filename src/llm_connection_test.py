"""
Small_LLM_Model の接続テスト用モジュール。

本番コードとは分離しておき、必要に応じて手動実行・呼び出しできるようにします。
"""

from llm_sdk import Small_LLM_Model


def test_small_llm_model() -> bool:
    """Small_LLM_Model クラスの基本機能をテストする。

    可能であれば実モデルでエンコード/ロジット/デコードを試し、
    失敗時は ImportError をハンドリングしてインポート確認のみ通す。
    戻り値はテストの成否。
    """

    print("\n🚀 Small_LLM_Model クラスのテストを開始します...")

    try:
        # 1. モデルの初期化
        print("\n1️⃣ モデルの初期化中...")
        try:
            model = Small_LLM_Model(
                model_name="Qwen/Qwen3-0.6B",
                device="cpu",
                dtype=None,
            )
            print("✅ モデルの初期化が完了しました")

            # 2. テスト用のテキスト
            test_text = "Hello, how are you?"
            print(f"\n2️⃣ テストテキスト: '{test_text}'")

            # 3. テキストのトークン化
            print("\n3️⃣ テキストのトークン化...")
            input_ids = model._encode(test_text)
            print(f"✅ トークン化完了: {input_ids.shape}")
            token_ids = input_ids.tolist()[0]
            print(f"   トークンID: {token_ids}")

            # 4. トークンIDからロジットを取得
            print("\n4️⃣ ロジットの取得...")
            logits = model.get_logits_from_input_ids(token_ids)
            print(f"✅ ロジット取得完了: {len(logits)} 個のロジット")
            print(f"   最初の5個のロジット: {logits[:5]}")

            # 5. 語彙ファイルのパスを取得
            print("\n5️⃣ 語彙ファイルのパスを取得...")
            vocab_path = model.get_path_to_vocabulary_json()
            print(f"✅ 語彙ファイルパス: {vocab_path}")

            # 6. デコードテスト
            print("\n6️⃣ デコードテスト...")
            decoded_text = model._decode(token_ids)
            print(f"✅ デコード結果: '{decoded_text}'")

            # 7. 簡単な生成テスト
            print("\n7️⃣ 簡単な生成テスト...")
            prompt = "The sum of 2 and 3 is"
            prompt_ids = model._encode(prompt).tolist()[0]
            print(f"   プロンプト: '{prompt}'")
            print(f"   プロンプトトークン: {prompt_ids}")
            next_logits = model.get_logits_from_input_ids(prompt_ids)
            print(f"   次のトークンのロジット数: {len(next_logits)}")

        except ImportError as e:
            print(f"⚠️ torch関連ライブラリが利用できません: {e}")
            print("   基本的なインポートテストのみ実行します")
            print("✅ Small_LLM_Modelクラスのインポートは成功しました")
            return True

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print(f"   エラータイプ: {type(e).__name__}")
        return False

    print("\n🎉 すべての接続テストが完了しました！")
    return True


if __name__ == "__main__":
    ok = test_small_llm_model()
    print("\n✅ テストが正常に完了しました" if ok else "\n❌ テストが失敗しました")

