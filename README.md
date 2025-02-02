# Advanced VRChat Avatar Creator - ボブにゃん版 🐱

高度な3Dモデリング機能を備えたVRChatアバター作成ツールです。PyQt6、PyOpenGL、その他の科学計算ライブラリを活用して、直感的なUIで高品質なアバターを作成できます。

## ✨ 主な機能

- 🎨 高度なメッシュ編集
  - キューブ追加
  - メッシュ分割
  - 変形
  - 頂点統合
  - 面押し出し
  - SciPyを利用したKDTree平滑化

- 🔧 詳細な編集ツール
  - 頂点エディタ
  - 変換行列エディタ
  - プロパティエディタ

- 📤 ファイル入出力
  - FBXエクスポート (Autodesk FBX SDK)
  - VRM入出力 (vrm2py)

- 🤖 VRChatアバター作成
  - ボーンリギング
  - ブレンドシェイプ
  - VRChat用最適化

- 🔌 プラグインシステム
  - カスタムプラグインのサポート
  - 動的プラグイン読み込み

## 🚀 インストール方法

1. 必要なパッケージのインストール:
```bash
pip install -r requirements.txt
```

2. 外部SDKのインストール:
- [Autodesk FBX SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3-1) (2020.3.1以降)
- [vrm2py](https://github.com/vrm-py/vrm2py)

## 📦 必要なパッケージ

- PyQt6
- PyOpenGL
- numpy
- scipy
- matplotlib
- plotly
- scikit-learn
- gltflib
- fbx
- vrm2py

## 💻 使用方法

1. プログラムの起動:
```bash
python "Advanced VRChat Avatar Creator.py"
```

2. 基本操作:
- メニューバーから「ファイル」→「キューブ追加」で新規メッシュ作成
- 各種編集ツールを使用してメッシュを編集
- 「VRChatアバター作成」メニューからアバターとして出力

3. マウス操作:
- 左クリック＋ドラッグ: モデル回転
- マウスホイール: ズーム

## 🎓 チュートリアル

プログラム内の「ヘルプ」→「インタラクティブチュートリアル」から、詳細な使用方法を学ぶことができます。

## 🔧 トラブルシューティング

- FBX SDKのエラー:
  - FBX SDKが正しくインストールされているか確認
  - 環境変数のパスが正しく設定されているか確認

- VRM出力エラー:
  - vrm2pyが正しくインストールされているか確認
  - 必要なボーン構造が正しく設定されているか確認

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

MITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 👥 作者

- ボブにゃん (Bob-nyaan)

## 📫 連絡先

- GitHub Issues: バグ報告や機能リクエスト
- Email: [your-email@example.com]

## 🌟 謝辞

- Autodesk FBX SDK
- vrm2pyプロジェクト
- PyQt6コミュニティ
- その他、このプロジェクトに貢献してくださった皆様

---
*注: このプロジェクトは開発中であり、機能は予告なく変更される可能性があります。* 