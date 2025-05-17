import os
import cv2
import numpy as np
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    Clusterer = hdbscan.HDBSCAN
    CLUSTERER_KW = dict(min_cluster_size=3, metric='euclidean')
    CLUSTERER_NAME = 'HDBSCAN'
except ImportError:
    Clusterer = DBSCAN
    CLUSTERER_KW = dict(eps=0.5, min_samples=1, metric='euclidean')
    CLUSTERER_NAME = 'DBSCAN'

def setup_logging(out_dir):
    log_file = os.path.join(out_dir, 'process.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ])

def detect_flags(img, out_dir):
    logging.info("STEP1: 旗領域の検出開始")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    roi_dir = os.path.join(out_dir, 'rois')
    os.makedirs(roi_dir, exist_ok=True)

    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        # アスペクト比と面積でフィルタ
        if 1.3 < w/h < 2.0 and w*h > 5000:
            roi = img[y:y+h, x:x+w]
            rois.append(roi)
            cv2.imwrite(os.path.join(roi_dir, f'flag_{i:03d}.png'), roi)
    logging.info(f"STEP1: 検出した旗ROI数 = {len(rois)}")
    return rois

def load_model(onnx_path):
    logging.info("STEP2: ONNXモデル読み込み")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    return net

def extract_features(net, rois, out_dir):
    logging.info("STEP3: 深層特徴ベクトル抽出開始")

    # 出力レイヤー名を取得（通常は一つだけ）
    out_names = net.getUnconnectedOutLayersNames()
    output_layer = out_names[0]
    logging.info(f"Using output layer: {output_layer}")

    feats = []
    for i, roi in enumerate(rois):
        blob = cv2.dnn.blobFromImage(
            roi, scalefactor=1/255.0,
            size=(224,224), mean=(0,0,0),
            swapRB=True, crop=True
        )
        net.setInput(blob)
        # 正しいレイヤー名を指定
        feat = net.forward(output_layer)   # → shape (1, 2048, 1, 1) など
        feat = feat.flatten()
        feats.append(feat)

    feats = np.vstack(feats)
    np.save(os.path.join(out_dir, 'features.npy'), feats)
    logging.info(f"STEP3: 抽出した特徴量 shape = {feats.shape}")
    return feats


def reduce_dim(feats, out_dir, max_components=512):
    n_samples, n_features = feats.shape
    # サンプル数か特徴次元数の小さいほう以下に調整
    n_components = min(n_samples, n_features, max_components)
    logging.info(f"STEP4: PCA 次元数を {n_components} に設定（要求上限: {max_components}）")
    pca = PCA(n_components=n_components, random_state=0)
    feats_pca = pca.fit_transform(feats)
    np.save(os.path.join(out_dir, 'features_pca.npy'), feats_pca)
    logging.info(
        f"STEP4: PCA後の shape = {feats_pca.shape}, "
        f"累積分散割合（上位5成分）= {pca.explained_variance_ratio_[:5].sum():.4f}"
    )
    return feats_pca




def cluster(feats, out_dir):
    logging.info(f"STEP5: {CLUSTERER_NAME} によるクラスタリング開始")
    clusterer = Clusterer(**CLUSTERER_KW)
    labels = clusterer.fit_predict(feats)
    # ノイズ(-1)を取り除いた実データクラスタ数
    unique = set(labels) - {-1}
    num_clusters = len(unique)
    # CSVで保存
    import pandas as pd
    df = pd.DataFrame({'roi_index': np.arange(len(labels)), 'label': labels})
    df.to_csv(os.path.join(out_dir, 'clusters.csv'), index=False)
    logging.info(f"STEP5: 検出クラスタ数 = {num_clusters} (ノイズを除く)")
    return labels, num_clusters

def main():
    parser = argparse.ArgumentParser(description="旗100種検出パイプライン")
    parser.add_argument('--image',   '-i', required=True, help="入力画像ファイル")
    parser.add_argument('--onnx',    '-m', required=True, help="ResNet50 ONNXモデル")
    parser.add_argument('--outdir',  '-o', default='output', help="出力ディレクトリ")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir)
    logging.info("=== パイプライン開始 ===")

    img = cv2.imread(args.image)
    rois = detect_flags(img, args.outdir)

    net = load_model(args.onnx)
    feats = extract_features(net, rois, args.outdir)

    # feats_pca = reduce_dim(feats, args.outdir, n_components=512)
    feats_pca = reduce_dim(feats, args.outdir, max_components=512)
    
    labels, num_clusters = cluster(feats_pca, args.outdir)

    # 判定
    if num_clusters >= 100:
        msg = "本当に100種以上の旗を検出できました！"
    else:
        msg = f"{num_clusters}種しか検出できませんでした…"
    logging.info("=== 判定結果: " + msg + " ===")

if __name__ == '__main__':
    main()
