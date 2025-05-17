import os
import cv2
import numpy as np
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, 'process.log'), mode='w')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[fh, logging.StreamHandler()]
    )


def detect_blocks(img, threshold_factor):
    """
    横方向の色変化を見てブロックを切り出しし、切り出し座標を返す
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    W = lab.shape[1]
    # 列ごとの Lab 平均色
    col_means = np.array([lab[:, x].mean(axis=0) for x in range(W)])
    # 隣接列間の距離
    diffs = np.linalg.norm(col_means[1:] - col_means[:-1], axis=1)
    thresh = diffs.mean() * threshold_factor
    boundaries = np.where(diffs > thresh)[0]
    cuts = np.concatenate([[0], boundaries + 1, [W]])
    return cuts.astype(int)


def extract_features(img, cuts, out_dir):
    """
    各ブロックの Lab 平均色を特徴量とし、切り出し保存
    """
    logging.info("STEP2: 特徴量抽出 (Lab平均色)")
    feats = []
    blk_dir = os.path.join(out_dir, 'blocks')
    os.makedirs(blk_dir, exist_ok=True)
    for i in range(len(cuts)-1):
        x0, x1 = cuts[i], cuts[i+1]
        blk = img[:, x0:x1]
        cv2.imwrite(os.path.join(blk_dir, f'block_{i:03d}.png'), blk)
        lab = cv2.cvtColor(blk, cv2.COLOR_BGR2LAB)
        mean3 = lab.mean(axis=(0,1))
        feats.append(mean3)
    feats = np.vstack(feats)
    np.savetxt(os.path.join(out_dir, 'features_lab.csv'), feats, delimiter=',')
    logging.info(f"STEP2: 特徴量行列 shape = {feats.shape}")
    return feats


def maybe_pca(feats, out_dir, var_threshold):
    if var_threshold <= 0:
        return feats
    logging.info("STEP3: PCA 次元圧縮開始")
    pca_full = PCA().fit(feats)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, var_threshold) + 1)
    logging.info(f"STEP3: 累積寄与率 {var_threshold*100:.1f}% に必要な次元数 = {n_comp}")
    pca = PCA(n_components=n_comp, random_state=0)
    feats_pca = pca.fit_transform(feats)
    np.save(os.path.join(out_dir, 'features_pca.npy'), feats_pca)
    logging.info(f"STEP3: PCA後の shape = {feats_pca.shape}")
    return feats_pca


def tune_dbscan(feats, out_dir, eps_min, eps_max, eps_steps, ms_min, ms_max):
    logging.info("STEP4: DBSCAN パラメータチューニング開始")
    best = (-1, None, None, None)
    records = []
    eps_vals = np.linspace(eps_min, eps_max, eps_steps)
    for eps in eps_vals:
        for ms in range(ms_min, ms_max+1):
            db = DBSCAN(eps=eps, min_samples=ms).fit(feats)
            labels = db.labels_
            n_clusters = len(set(labels) - {-1})
            if n_clusters < 2:
                continue
            score = silhouette_score(feats, labels)
            records.append((eps, ms, n_clusters, score))
            if score > best[0]:
                best = (score, eps, ms, n_clusters)
    df = pd.DataFrame(records, columns=['eps','min_samples','n_clusters','silhouette'])
    df.to_csv(os.path.join(out_dir, 'dbscan_tuning.csv'), index=False)
    if best[1] is None:
        logging.warning("適切なパラメータが見つかりませんでした。既定値を使用します。")
        return (0.0, eps_min, ms_min, 0)
    logging.info(f"STEP4: 最適パラメータ → eps={best[1]:.3f}, min_samples={best[2]}, クラスタ数={best[3]}, silhouette={best[0]:.3f}")
    return best


def cluster_and_annotate(img, cuts, feats, eps, ms, out_dir):
    """
    DBSCAN を実行し、オリジナル画像に境界線とクラスタIDを描画して保存
    """
    logging.info("STEP5: DBSCAN クラスタリング＆可視化開始")
    db = DBSCAN(eps=eps, min_samples=ms).fit(feats)
    labels = db.labels_
    n_clusters = len(set(labels) - {-1})
    # 描画用コピー
    vis = img.copy()
    # 色・フォント設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(cuts)-1):
        x0, x1 = cuts[i], cuts[i+1]
        # 境界線
        cv2.rectangle(vis, (x0, 0), (x1, img.shape[0]), (0,0,0), 2)
        # ラベルを表示
        clu = labels[i]
        text = str(clu if clu!=-1 else 'noise')
        cv2.putText(vis, text, (x0+5, 30), font, 1.0, (255,255,255), 2)
    # 保存
    out_path = os.path.join(out_dir, 'result.png')
    cv2.imwrite(out_path, vis)
    logging.info(f"STEP5: 可視化結果を保存: {out_path}")
    return labels, n_clusters


def main():
    parser = argparse.ArgumentParser(description="色ブロック数カウントパイプライン (画像保存付き)")
    parser.add_argument('-i','--image', required=True, help="切り抜き済み旗画像")
    parser.add_argument('-o','--outdir', default='output', help="出力ディレクトリ")
    parser.add_argument('--thresh_fac', type=float, default=1.2, help="列間距離閾値係数")
    parser.add_argument('--pca_var', type=float, default=0.90, help="PCA 累積寄与率閾値 (0 で無効)")
    parser.add_argument('--eps_min', type=float, default=5.0)
    parser.add_argument('--eps_max', type=float, default=20.0)
    parser.add_argument('--eps_steps', type=int, default=16)
    parser.add_argument('--ms_min', type=int, default=1)
    parser.add_argument('--ms_max', type=int, default=5)
    args = parser.parse_args()

    setup_logging(args.outdir)
    logging.info("=== パイプライン開始 ===")

    img = cv2.imread(args.image)
    cuts = detect_blocks(img, args.thresh_fac)
    feats = extract_features(img, cuts, args.outdir)
    feats_reduced = maybe_pca(feats, args.outdir, args.pca_var)

    _, eps, ms, _ = tune_dbscan(feats_reduced, args.outdir,
                                 args.eps_min, args.eps_max,
                                 args.eps_steps, args.ms_min, args.ms_max)
    labels, final_cnt = cluster_and_annotate(img, cuts, feats_reduced, eps, ms, args.outdir)

    msg = f"最終的に {final_cnt} 種のブロックを検出しました。"
    logging.info("=== 判定結果: " + msg + " ===")

if __name__ == '__main__':
    main()

