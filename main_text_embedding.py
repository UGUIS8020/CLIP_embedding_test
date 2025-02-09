import os
import re
import glob
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 環境変数をロード
load_dotenv()

def validate_environment():
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

def extract_main_text(file_path):
    """テキストから本文を抽出（[Fig]より前の部分）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        main_text = text_data.split('[Fig')[0].strip()
        return main_text
    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        return ""

def initialize_services():
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index("text-search")
    return client, index

def main():
    try:
        client, index = initialize_services()
        
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")

        vectors_to_upsert = []
        for file_path in txt_files:
            main_text = extract_main_text(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # テキストのチャンク分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", "、", " ", ""]
            )
            chunks = text_splitter.split_text(main_text)
            print(f"{file_path}: {len(chunks)}個のチャンクを作成")

            # 各チャンクの処理
            for i, chunk in enumerate(chunks):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    chunk_embedding = response.data[0].embedding
                    
                    vectors_to_upsert.append((
                        f"{base_name}_chunk_{i}",
                        chunk_embedding,
                        {
                            "type": "main_text",
                            "text": chunk,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source_file": base_name,
                            "category": "dental"
                        }
                    ))
                    print(f"- チャンク {i+1}/{len(chunks)} の処理完了")
                except Exception as e:
                    print(f"チャンク {i} の処理中にエラー: {e}")

        # Pineconeへのアップロード
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"\n{len(vectors_to_upsert)}個のチャンクをPineconeに保存しました")
        else:
            print("\nアップロードするデータがありません")

    except Exception as e:
        print(f"処理中にエラーが発生: {e}")

if __name__ == "__main__":
    main()