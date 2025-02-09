import os
import re
import glob
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
            image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        # 512次元を1536次元に拡張
        vector = image_embedding.cpu().numpy().tolist()[0]
        expanded_vector = np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
        return {
            "vector": expanded_vector,
            "model": "open_clip",
            "status": "success"
        }
    except Exception as e:
        print(f"画像エンベディングエラー: {e}")
        return {
            "vector": [0.0] * 1536,  # 1536次元のゼロベクトル
            "model": "open_clip",
            "status": "error",
            "error_message": str(e)
        }

# 512次元ベクトルを 1536次元に拡張（ゼロ埋め）
def expand_embedding(embedding_dict, target_dim=1536):
    vector = embedding_dict["vector"]
    extra_dims = target_dim - len(vector)
    expanded_vector = np.concatenate([vector, np.zeros(extra_dims)]).tolist()
    
    return {
        "vector": expanded_vector,
        "original_dim": len(vector),
        "model": embedding_dict["model"],
        "status": embedding_dict["status"]
    }

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
        client, index, model, preprocess, device = initialize_services()
        
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")
        
        all_fig_texts = {}
        for file_path in txt_files:
            fig_texts = extract_fig_texts(file_path)
            all_fig_texts.update(fig_texts)
            print(f"{file_path}: {len(fig_texts)}個の [FigX] 説明文を取得")

        vectors_to_upsert = []
        for fig, text in all_fig_texts.items():
            # ベース名を取得（例：Fig1）
            base_id = fig  # 既にFig1などの形式

            # テキストのembedding取得とアップロード
            text_emb = get_text_embedding(text, client)
            vectors_to_upsert.append((
                base_id,  # Fig1
                text_emb,
                {
                    "type": "text",
                    "text": text,
                    "related_image": f"{base_id}.jpg",  # Fig1.jpg
                    "category": "dental",  
                    "chapter": "chapter2",  # 章の情報を追加
                    "chapter_title": "歯と歯周組織の発生と解剖",  # 章タイトル
                    "fig_id": f"{base_name}_{base_id}"  # 例: chapter2_Fig1                  
                }
            ))
            
             # 画像の処理
            image_path = f"data/{base_id}.jpg"
            if os.path.exists(image_path):
                image_result = get_image_embedding(image_path, model, preprocess, device)
                if image_result["status"] == "success":
                    image_emb = image_result["vector"]
                    vectors_to_upsert.append((
                        f"{base_id}.jpg",  # Fig1.jpg
                        image_emb,
                        {
                            "type": "image",
                            "text": text,
                            "related_text": base_id,  # Fig1
                            "category": "dental", 
                            "chapter_title": "歯と歯周組織の発生と解剖",
                            "fig_id": f"{base_name}_{base_id}"                           
                        }
                    ))
            print(f"- {fig}の処理完了")

        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"\n{len(vectors_to_upsert)}個のベクトルをPineconeに保存しました")
            except Exception as e:
                print(f"\nPineconeへの保存中にエラーが発生: {e}")
        else:
            print("\nアップロードするデータがありません")

    except Exception as e:
        print(f"\n処理中にエラーが発生: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()