import os
import re
import numpy as np
import openai
import torch
import open_clip
import traceback
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from pathlib import Path

def validate_environment():
    """環境変数の検証"""
    load_dotenv()
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

def initialize_services():
    """サービスの初期化"""
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index("text-search")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    returned_values = open_clip.create_model_and_transforms(
        "ViT-B/32",
        pretrained="openai",
        jit=True,
        force_quick_gelu=True
    )
    
    if len(returned_values) == 2:
        model, preprocess = returned_values
    elif len(returned_values) == 3:
        model, preprocess, _ = returned_values
    else:
        raise ValueError("open_clip.create_model_and_transforms からの予期しない戻り値の数です")
    
    model.to(device)
    return client, index, model, preprocess, device

def extract_and_merge_fig_texts(file_path):
    """テキストから全てのFigure説明を抽出し、関連する説明を統合する"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        # 全てのFigure説明を抽出
        pattern = r"\[Fig(\d+[a-z]?)\]\s*(.*?)(?=\n\[Fig|$)"
        matches = re.findall(pattern, text_data, re.DOTALL)
        
        # Figureごとに説明文を整理
        fig_groups = {}
        for num, desc in matches:
            # ベースとなるFigure番号を抽出（例：Fig1a → Fig1）
            base_fig = re.match(r'(\d+)', num).group(1)
            base_key = f"Fig{base_fig}"
            
            # 説明文を整形
            cleaned_desc = desc.strip()
            
            # ベースFigureごとにグループ化
            if base_key not in fig_groups:
                fig_groups[base_key] = {
                    'main_text': [],
                    'sub_figs': {},
                    'full_id': f"Fig{num}"
                }
            
            # サブフィギュアがある場合（Fig1a, Fig1b など）
            if len(num) > len(base_fig):
                sub_key = f"Fig{num}"
                fig_groups[base_key]['sub_figs'][sub_key] = cleaned_desc
            else:
                fig_groups[base_key]['main_text'].append(cleaned_desc)

        # 最終的な説明文を構築
        final_texts = {}
        for base_key, group in fig_groups.items():
            combined_text = []
            
            # メインの説明文を追加
            if group['main_text']:
                combined_text.extend(group['main_text'])
            
            # サブフィギュアの説明文を追加
            if group['sub_figs']:
                sub_text = [f"{sub_id}: {desc}" for sub_id, desc in sorted(group['sub_figs'].items())]
                combined_text.extend(sub_text)
            
            # 最終テキストを作成
            final_texts[base_key] = " ".join(combined_text)

        print(f"\n📚 {len(final_texts)}個のFigure説明を抽出・統合しました")
        for fig_id, text in final_texts.items():
            print(f"\n✅ {fig_id}の説明文を統合:")
            print(f"   長さ: {len(text)}文字")
            print(f"   プレビュー: {text[:100]}...")
        
        return final_texts

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

def check_image_files(fig_texts):
    """画像ファイルの存在を確認"""
    data_dir = Path("data")
    missing_images = []
    existing_images = []
    
    for fig_id in fig_texts.keys():
        image_path = data_dir / f"{fig_id}.jpg"
        if not image_path.exists():
            missing_images.append(fig_id)
        else:
            existing_images.append(fig_id)
    
    if missing_images:
        print("\n⚠️ 以下の画像ファイルが見つかりません:")
        for fig_id in missing_images:
            print(f"  - data/{fig_id}.jpg")
    
    if existing_images:
        print("\n✅ 以下の画像ファイルが利用可能です:")
        for fig_id in existing_images:
            print(f"  - data/{fig_id}.jpg")
    
    return existing_images, missing_images

def get_image_embedding(image_path, model, preprocess, device):
    """画像のエンベディングを取得"""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image).cpu().numpy().flatten()
        return image_embedding
    except Exception as e:
        print(f"画像エンベディングエラー ({image_path}): {e}")
        return np.zeros(512)

def get_text_embedding(text, client):
    """テキストのエンベディングを取得"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"テキストエンベディングエラー: {e}")
        return np.zeros(1536)

def main():
    try:
        # サービスの初期化
        client, index, model, preprocess, device = initialize_services()
        
        file_path = "data/chapter2_test.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        # 既存のベクトルを削除
        index.delete(delete_all=True)
        print("🗑️ 既存のベクトルを削除しました")
        
        # Figure説明の抽出と統合
        fig_texts = extract_and_merge_fig_texts(file_path)
        
        # 画像ファイルの確認
        existing_images, missing_images = check_image_files(fig_texts)

        # Pineconeへのアップロード用データの準備
        to_upsert = []
        for fig_id, text in fig_texts.items():
            # ベースとなるFigure IDを使用
            base_fig_id = fig_id
            
            # テキストエンベディング
            text_emb = get_text_embedding(text, client)
            
            # 画像エンベディング（利用可能な場合）
            combined_emb = text_emb
            image_paths = []
            
            # メインの画像をチェック
            main_image_path = f"data/{base_fig_id}.jpg"
            if os.path.exists(main_image_path):
                image_paths.append(main_image_path)
            
            # サブフィギュアの画像をチェック
            if any(c.isdigit() for c in base_fig_id):
                base_num = ''.join(filter(str.isdigit, base_fig_id))
                for suffix in ['a', 'b', 'c', 'd']:
                    sub_image_path = f"data/{base_fig_id}{suffix}.jpg"
                    if os.path.exists(sub_image_path):
                        image_paths.append(sub_image_path)
            
            # 画像が存在する場合、エンベディングを結合
            if image_paths:
                image_embeddings = []
                for img_path in image_paths:
                    img_emb = get_image_embedding(img_path, model, preprocess, device)
                    img_emb = np.tile(img_emb, 1536 // len(img_emb))
                    image_embeddings.append(img_emb)
                
                if image_embeddings:
                    # 全ての画像エンベディングの平均を取る
                    avg_image_emb = np.mean(image_embeddings, axis=0)
                    # テキストと画像のエンベディングを結合
                    combined_emb = np.mean([text_emb, avg_image_emb], axis=0)
            
            # メタデータの準備
            metadata = {
                "figure_id": base_fig_id,
                "text": text
            }
            if image_paths:
                metadata["image_paths"] = image_paths
            
            # ゼロベクトルでないことを確認
            if not np.allclose(combined_emb, np.zeros_like(combined_emb)):
                to_upsert.append((base_fig_id, combined_emb.tolist(), metadata))

        # Pineconeへのアップロード
        if to_upsert:
            index.upsert(vectors=to_upsert)
            print(f"\n✨ {len(to_upsert)}個のベクトルをPineconeに保存しました")
            
            # 保存されたデータの概要を表示
            print("\n📊 保存されたデータの概要:")
            for vector_id, _, metadata in to_upsert:
                print(f"\n🔹 {vector_id}:")
                print(f"   テキスト長: {len(metadata['text'])}文字")
                if 'image_paths' in metadata:
                    print(f"   関連画像: {len(metadata['image_paths'])}枚")
        else:
            print("\n⚠️ アップロードするデータがありません")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()