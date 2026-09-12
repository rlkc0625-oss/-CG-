# 図面→内観CGメーカー 無料Ver.1

スマホから使えるStreamlitアプリの試作版です。
AI Hordeの無料・匿名APIを利用してimg2imgを試します。

## 公開方法
GitHubに `app.py` と `requirements.txt` をアップロードし、
Streamlit Community Cloudから `app.py` を指定してデプロイします。

## 注意
AI Hordeは無料のコミュニティGPUサービスです。匿名リクエストは混雑時に優先度が低くなります。
また、img2imgなので図面の壁・窓・ドア等を完全にCAD精度で保持するものではありません。
