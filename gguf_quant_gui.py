import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import threading

llama_cpp_path=""#ここに各種変換プログラムのパスを指定

# 量子化オプションのリスト
quantization_types = [
    "Q2_K", "Q2_K_S", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS",
    "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"
]
error_type=0 #エラー時の処理分岐フラグ

def select_hf_model():
    file_path = filedialog.askdirectory(title="HuggingFaceモデルのディレクトリ")
    hf_model_entry.delete(0, tk.END)
    hf_model_entry.insert(0, file_path)

def select_imatrix_file():
    file_path = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="imatrixデータセット")
    imatrix_entry.delete(0, tk.END)
    imatrix_entry.insert(0, file_path)

def select_output_gguf():
    file_path = filedialog.asksaveasfilename(defaultextension=".gguf", filetypes=[("GGUF files", "*.gguf")], title="BF16 GGUFファイル")
    output_gguf_entry.delete(0, tk.END)
    output_gguf_entry.insert(0, file_path)

def select_output_imatrix():
    file_path = filedialog.asksaveasfilename(defaultextension=".imatrix", filetypes=[("IMatrix files", "*.imatrix")], title="imatrixファイル")
    output_imatrix_entry.delete(0, tk.END)
    output_imatrix_entry.insert(0, file_path)

def select_output_quantized():
    file_path = filedialog.asksaveasfilename(defaultextension=".gguf", filetypes=[("GGUF files", "*.gguf")], title="量子化GGUFファイル")
    output_quantized_entry.delete(0, tk.END)
    output_quantized_entry.insert(0, file_path)

def update_status(message):
    status_label.config(text=message)

def run_conversion():
    def _run_conversion():
        global error_type
        hf_model = hf_model_entry.get()
        imatrix_file = imatrix_entry.get()
        output_gguf = output_gguf_entry.get()
        output_imatrix = output_imatrix_entry.get()
        output_quantized = output_quantized_entry.get()
        quantization_type = quantization_type_var.get()  # 量子化オプションのリスト選択した値を取得

        if not all([output_gguf, output_quantized]) or (imatrix_file and not output_imatrix):
            update_status("Error: 必要なフィールドを入力してください")
            messagebox.showerror("Error", "必要なフィールドを入力してください")
            return

        if imatrix_file and not os.path.isfile(imatrix_file):
            update_status("Error: 選択された imatrix データセットファイルが存在しません")
            messagebox.showerror("Error", "選択された imatrix データセットファイルが存在しません")
            return
        
        if not os.path.exists(output_imatrix) and quantization_type in["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS","IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ3_XS", "IQ4_NL", "IQ4_XS"]:
           update_status("Error: imatrix量子化に必要なimatrixファイルが指定されていません")
           messagebox.showerror("Error", "imatrix量子化に必要なimatrixファイルが指定されていません")
           return
        try:
            update_status("HFからBF16 GGUFへ変換しています…")
            # HFからGGUFに変換
            if not os.path.exists(output_gguf)and hf_model:
                process = subprocess.Popen(
                    ['python', f'{llama_cpp_path}\\convert_hf_to_gguf.py', hf_model, '--outtype', 'bf16', '--outfile', output_gguf],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding='utf-8'  # UTF-8 でエンコード
                )
                capture_output(process, "HF to GGUF Conversion")
            else:
                #update_status( f"{output_gguf} は既に存在します。この処理をスキップします")
                #messagebox.showinfo("Info", f"{output_gguf} は既に存在します。この処理をスキップします。")
                pass

            update_status("imatrixの生成をしています…")
            # imatrixの作成（imatrix_fileが提供されている場合のみ）
            if imatrix_file and not os.path.exists(output_imatrix):
                process = subprocess.Popen(
                    [f'{llama_cpp_path}\\llama-imatrix', '-m', output_gguf, '-f', imatrix_file, '-o', output_imatrix, '--chunks', '32'],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding='utf-8'  # UTF-8 でエンコード
                )
                capture_output(process, "IMatrix Creation")
            elif imatrix_file:
                #update_status(f"{output_imatrix} は既に存在します。この処理をスキップします")
                #messagebox.showinfo("Info", f"{output_imatrix} は既に存在します。この処理をスキップします。")
                pass

            update_status(f"モデルを{quantization_type}で量子化しています…")
            # モデルを量子化
            if  os.path.exists(output_gguf) and not os.path.exists(output_quantized):
                # imatrixを使用するとき
                if os.path.exists(output_imatrix):
                    update_status(f"モデルをimatrixあり{quantization_type}で量子化しています…")
                    # Use the imatrix file if provided
                    process = subprocess.Popen(
                        [f'{llama_cpp_path}\\llama-quantize', '--imatrix', output_imatrix, output_gguf, output_quantized, quantization_type],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True,
                        encoding='utf-8'  # UTF-8 でエンコード
                    )
                    capture_output(process, "Quantization with imatrix")
                else:
                    # imatrix不使用のとき
                    process = subprocess.Popen(
                        [f'{llama_cpp_path}\\llama-quantize', output_gguf, output_quantized, quantization_type],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True,
                        encoding='utf-8'  # UTF-8 でエンコード
                    )
                    capture_output(process, "Quantization")
            if  os.path.exists(output_quantized):
                #update_status(f"{output_quantized} は既に存在します。この処理をスキップします")
                #messagebox.showinfo("Info", f"{output_quantized} は既に存在します。この処理をスキップします。")
                pass
        except subprocess.CalledProcessError as e :
            error_type=1
            update_status(f"Error: プロセスが失敗しました")
            messagebox.showerror("Error", f"プロセスが失敗しました: {e}")
        except FileNotFoundError as e:
            error_type=1
            update_status("Error: 必要なllama.cpp関連ファイルが見つかりません")
            messagebox.showerror("Error", " 必要なllama.cpp関連ファイルが見つかりません")
        except ValueError as e:
            error_type=1
            update_status(f"Error: プロセスが失敗しました: {e}")
            messagebox.showerror("Error", f"プロセスが失敗しました:")
        if os.path.exists(output_quantized) and error_type==0:
           update_status("変換と量子化が完了しました")
           messagebox.showinfo("Success", "変換と量子化が完了しました")
        if not os.path.exists(output_quantized) or error_type==1:
           error_type=0
           update_status(f"Error: 変換と量子化に失敗しました ")
           messagebox.showerror("Error", f"変換と量子化に失敗しました:\n各フィールドが正常か確認してください ")
        if error_type==2:
           error_type=0
    threading.Thread(target=_run_conversion).start()

process_list = []
def capture_output(process, process_name):
    process_list.append(process)  # プロセスをリストに追加
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[{process_name}] {output.strip()}")
        if output.strip().find("main: failed")!=-1:
            raise ValueError("process failed")
    rc = process.poll()
    try:
     process_list.remove(process)  # プロセスが終了したらリストから削除
    except:
     pass
    return rc

def on_closing():
    for process in process_list:
        process.terminate()  # すべてのプロセスを強制終了
    root.destroy()

def process_stop():
    global process_list, error_type
    if process_list :
     for process in process_list:
         process.terminate()  # すべてのプロセスを強制終了
     process_list = []
     error_type=2
     update_status(f"Error: 変換と量子化を強制終了しました")
     messagebox.showerror("Error", f"変換と量子化を強制終了しました")
    else:
       pass
    return

# メインウィンドウの作成
root = tk.Tk()
root.title("gguf_quant_gui")

# ウィジェットの配置と作成
tk.Label(root, text="HuggingFaceモデルのディレクトリ:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
hf_model_entry = tk.Entry(root, width=50)
hf_model_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="開く", command=select_hf_model).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="imatrixデータセット:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
imatrix_entry = tk.Entry(root, width=50)
imatrix_entry.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="開く", command=select_imatrix_file).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="BF16 GGUFファイルの保存先:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
output_gguf_entry = tk.Entry(root, width=50)
output_gguf_entry.grid(row=2, column=1, padx=10, pady=5)
tk.Button(root, text="名前を付けて保存", command=select_output_gguf).grid(row=2, column=2, padx=10, pady=5)

tk.Label(root, text="imatrixファイルの保存先:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
output_imatrix_entry = tk.Entry(root, width=50)
output_imatrix_entry.grid(row=3, column=1, padx=10, pady=5)
tk.Button(root, text="名前を付けて保存", command=select_output_imatrix).grid(row=3, column=2, padx=10, pady=5)

tk.Label(root, text="量子化GGUFファイルの保存先:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
output_quantized_entry = tk.Entry(root, width=50)
output_quantized_entry.grid(row=4, column=1, padx=10, pady=5)
tk.Button(root, text="名前を付けて保存", command=select_output_quantized).grid(row=4, column=2, padx=10, pady=5)

tk.Label(root, text="量子化タイプ:").grid(row=5, column=0, padx=10, pady=5, sticky='w')
quantization_type_var = tk.StringVar(root)
quantization_type_var.set("Q4_K_M")  # デフォルトの選択肢
quantization_type_menu = tk.OptionMenu(root, quantization_type_var, *quantization_types)
quantization_type_menu.grid(row=5, column=1, padx=10, pady=5)

# ステータスラベル
status_label = tk.Label(root, text="待機中…", width=50, anchor='w')
status_label.grid(row=6, column=0, columnspan=3, padx=10, pady=5)

# 実行ボタン
run_button = tk.Button(root, text="変換と量子化を実行", command=run_conversion)
run_button.grid(row=7, column=0, columnspan=3, pady=20)
stop_button = tk.Button(root, text="量子化を中止", command=process_stop)
stop_button.grid(row=7, column=2, pady=20)

# WM_DELETE_WINDOW プロトコルのハンドラを設定
root.protocol("WM_DELETE_WINDOW", on_closing)
# GUIを開始
root.mainloop()