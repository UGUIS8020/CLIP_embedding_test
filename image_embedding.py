import os
import numpy as np
import torch
import open_clip
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone

# 環境変数のロード
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pinecone 初期化
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("text-search")  # インデックスの参照

# CLIP モデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B/32", pretrained="openai")
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
model.to(device)

# 画像フォルダの設定
image_folder = "./data/"

# 画像を CLIP でエンベディング化
def get_image_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy().tolist()[0]  # 512次元ベクトル
    except Exception as e:
        print(f"画像エンベディングエラー: {e}")
        return [0.0] * 512  # エラー時はゼロベクトルを返す

# 512次元ベクトルを 1536次元に拡張（ゼロ埋め or ランダムノイズ追加）
def expand_embedding(embedding, target_dim=1536):
    extra_dims = target_dim - len(embedding)
    noise = np.random.normal(0, 0.01, extra_dims)  # 平均0, 標準偏差0.01のノイズを追加
    return np.concatenate([embedding, noise]).tolist()

# 画像の処理とアップロード
def process_and_upload_images():
    print("\n画像データの処理を開始...")
    image_vectors = []

    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)

            # 画像の埋め込み取得
            image_embedding = get_image_embedding(image_path)

            # 512次元を 1536次元に拡張
            expanded_embedding = expand_embedding(image_embedding)

            # メタデータの作成
            metadata = {"image": filename}

            # Pinecone にアップロードするデータリストに追加
            image_vectors.append((filename, expanded_embedding, metadata))
            print(f"画像埋め込み作成: {filename}")

    # デバッグ: Pinecone にアップロードするデータを確認
    print("\nアップロードするデータ:")
    for filename, vector, metadata in image_vectors[:3]:  # 先頭3つを表示
        print(f"ID: {filename}")
        print(f"ベクトル次元: {len(vector)}")
        print(f"メタデータ: {metadata}")
        print(f"ベクトルの一部: {vector[:10]}")
        print("-" * 50)

    # Pinecone に一括アップロード
    if image_vectors:
        index.upsert(image_vectors)
        print("\nすべての画像データを Pinecone にアップロード完了！")
    else:
        print("\n画像が見つかりませんでした。")

# 実行
if __name__ == "__main__":
    process_and_upload_images()