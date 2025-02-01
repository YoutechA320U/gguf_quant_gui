## gguf_quant_gui

ダウンロード済のHuggingFaceのモデルを一定のルールでgguf量子化するGUIツール。


## 使い方
* gguf_quant_gui.pyのllama_cpp_path=""にllama.cppをCUIで実行する際のディレクトリを記述します。

* 実行してGUIが開いたら必要なフィールドを入力します。

* 入力が完了したら"Run Conversion and Quantization"を押すと量子化が開始します。

* 変換が完了したら "すべてのプロセスが成功しました。"　のメッセージウィンドウが表示されます。

## 備考

* HuggingFaceモデルからgguf変換する際はf16で変換します。

* imatrixはchunks 32で作成します。

* imatrixのデータセットファイルを入力しない場合はimatrixなしで量子化します。

* 出力先のファイルが存在する場合、"～ は既に存在します。このステップをスキップします。"のメッセージボックスが表示され出力がスキップされます。

* 必要なフィールドの入力が足りない場合は"必要なフィールドを入力してください。"のエラーメッセージが表示されます

* 量子化に失敗した場合は"プロセスが失敗しました: "のエラーメッセージが表示されます。

* 量子化中GUIが応答なしになるのは仕様です。

※本ツールは[Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)によって生成されました。

## 履歴
    [2025/11/13] - 初回リリース