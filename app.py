from dotenv import load_dotenv
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
import uuid
from pathlib import Path
from typing import List, Dict
import re
import glob

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split('[。!？\n\r]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_idx': len(''.join(chunks).strip()),
                    'length': len(chunk_text)
                })
                
                overlap_size = 0
                overlap_chunks = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) > self.overlap:
                        break
                    overlap_chunks.insert(0, s)
                    overlap_size += len(s)
                current_chunk = overlap_chunks
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_idx': len(''.join(chunks).strip()),
                'length': len(chunk_text)
            })

        return chunks

class PineconeStore:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        load_dotenv()
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "text-search"
        self.index = self.pc.Index(self.index_name)
        
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    def get_text_embedding(self, text_content: str) -> torch.Tensor:
        text_inputs = self.processor(text=text_content, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**text_inputs)
        return text_embedding.cpu().numpy()[0]

    def get_image_embedding(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()[0]

    def process_single_document(self, text_file: str, image_files: List[str]) -> Dict:
        """単一のドキュメントを処理"""
        text_path = Path(text_file)
        image_paths = [Path(img_file) for img_file in image_files]
        
        # ファイルの存在確認
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        for img_path in image_paths:
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

        # テキストの読み込みとチャンク分割
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        chunks = self.chunker.create_chunks(text_content)
        doc_id = str(uuid.uuid4())
        
        # チャンクごとにベクトルを保存
        for chunk_idx, chunk in enumerate(chunks):
            chunk_embedding = self.get_text_embedding(chunk['text'])
            self.index.upsert(
                vectors=[{
                    'id': f"{doc_id}_text_{chunk_idx}",
                    'values': chunk_embedding.tolist(),  # ここでリストに変換
                    'metadata': {
                        'type': 'text',
                        'file_name': text_file,
                        'content': chunk['text'],
                        'doc_id': doc_id,
                        'chunk_idx': chunk_idx,
                        'start_idx': chunk['start_idx'],
                        'length': chunk['length'],
                        'total_chunks': len(chunks),
                        'related_images': image_files
                    }
                }]
            )
        
        # 画像の処理と保存
        for idx, img_path in enumerate(image_paths):
            image_embedding = self.get_image_embedding(img_path)
            self.index.upsert(
                vectors=[{
                    'id': f"{doc_id}_image_{idx}",
                    'values': image_embedding.tolist(),
                    'metadata': {
                        'type': 'image',
                        'file_name': img_path.name,
                        'doc_id': doc_id,
                        'parent_text': text_file,
                        'image_index': idx
                    }
                }]
            )
        
        return {
            'doc_id': doc_id,
            'text_file': text_file,
            'image_files': image_files,
            'num_chunks': len(chunks)
        }

    def store_document(self, text_pattern: str, image_files: List[str]) -> List[Dict]:
        """dataフォルダー内のファイルを処理"""
        # dataフォルダー内のファイルを検索
        data_dir = Path('data')
        if not data_dir.exists():
            print(f"エラー: {data_dir} フォルダーが見つかりません")
            return []

        # テキストファイルの検索
        text_files = list(data_dir.glob(text_pattern))
        if not text_files:
            print(f"警告: data/{text_pattern} に一致するテキストファイルが見つかりません")
            return []

        # 画像ファイルのパスを修正
        image_paths = [data_dir / img for img in image_files]
        
        # 画像ファイルの存在確認
        missing_images = [img for img in image_paths if not img.exists()]
        if missing_images:
            print(f"警告: 以下の画像ファイルが見つかりません:")
            for img in missing_images:
                print(f"- {img}")
            return []
        
        results = []
        print("\n処理を開始します...")
        for text_file in text_files:
            print(f"\n{text_file.name} の処理中:")
            try:
                # 相対パスを使用して処理
                result = self.process_single_document(
                    str(text_file),
                    [str(img_path) for img_path in image_paths]
                )
                results.append(result)
                print(f"✓ {text_file.name} の処理が完了しました")
            except Exception as e:
                print(f"× エラー ({text_file.name}): {str(e)}")
        
        return results

def main():
    # ストアの初期化
    store = PineconeStore(
        chunk_size=1000,
        overlap=200
    )
    
    print("Pineconeストアを初期化しました")
    
    # data フォルダー内のファイルを処理
    results = store.store_document(
        "*.txt",  # txtファイルを処理
        ["Fig1.jpg", "Fig2.jpg", "Fig3.jpg"]
    )
    
    # 結果の表示
    if results:
        print("\n処理サマリー:")
        for result in results:
            print(f"\nドキュメントID: {result['doc_id']}")
            print(f"テキストファイル: {result['text_file']}")
            print(f"チャンク数: {result['num_chunks']}")
            print(f"関連画像: {', '.join(result['image_files'])}")
    else:
        print("\n処理するファイルがないか、エラーが発生したため処理を完了できませんでした。")

if __name__ == "__main__":
    main()