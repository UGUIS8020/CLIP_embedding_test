import os
import re
import numpy as np
import openai
import pinecone
import torch
import open_clip
import traceback
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 環境変数をロード
load_dotenv()

# 環境変数の検証
def extract_fig_texts(file_path):
    """テキストから [FigX] の説明を抽出"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        pattern = r"\[Fig(\d+[a-z]?)\]\s*(.*?)(?=\n\[Fig|$)"
        matches = re.findall(pattern, text_data, re.DOTALL)
        return {f"Fig{num}": desc.strip() for num, desc in matches}

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

def validate_environment():
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

# 初期化処理をまとめる
def initialize_services():
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index("text-search")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    returned_values = open_clip.create_model_and_transforms(
        "ViT-B/32",
        pretrained="openai",
        jit=True,  # JITコンパイルを有効化
        force_quick_gelu=True  # QuickGELUを強制的に有効化
    )
    
    if len(returned_values) == 2:
        model, preprocess = returned_values
    elif len(returned_values) == 3:
        model, preprocess, _ = returned_values
    else:
        raise ValueError("open_clip.create_model_and_transforms からの予期しない戻り値の数です")
    
    model.to(device)
    return client, index, model, preprocess, device

def get_image_embedding(image_path, model, preprocess, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image).cpu().numpy().flatten()
        return image_embedding
    except Exception as e:
        print(f"画像エンベディングエラー: {e}")
        return np.zeros(512)  # None の代わりにゼロベクトルを返す

def get_text_embedding(text, client, chunk_size=1000, chunk_overlap=200):
    if isinstance(text, dict):
        text = ' '.join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # 逐次的に平均を計算
    embedding_sum = np.zeros(1536)
    count = 0
    
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding_sum += np.array(response.data[0].embedding)
            count += 1
        except Exception as e:
            print(f"Embedding生成エラー: {e}")
            print(traceback.format_exc())
    
    return (embedding_sum / count if count > 0 else embedding_sum).tolist()

def main():
    try:
        # サービスの初期化
        client, index, model, preprocess, device = initialize_services()
        
        file_path = "data/chapter2_test.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        fig_texts = extract_fig_texts(file_path)
        print(f"{len(fig_texts)}個の [FigX] 説明文を取得")

        # テキストとイメージの埋め込みを処理
        embeddings = {}
        for fig, text in fig_texts.items():
            text_emb = get_text_embedding(text, client)
            
            image_path = f"data/{fig}.jpg"
            if os.path.exists(image_path):
                image_emb = get_image_embedding(image_path, model, preprocess, device)
                image_emb = np.tile(image_emb, 1536 // len(image_emb))
            else:
                image_emb = np.zeros(1536)
            
            embeddings[fig] = np.mean([text_emb, image_emb], axis=0).tolist()

        # Pineconeへのアップロード
        to_upsert = [
            (fig, vector, {"text": fig_texts[fig]})
            for fig, vector in embeddings.items()
            if not np.allclose(vector, np.zeros_like(vector))
        ]

        if to_upsert:
            index.upsert(vectors=to_upsert)
            print("Pineconeにデータを保存しました（1536次元で統一済み）")
        else:
            print("アップロードするデータがありません")

    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()