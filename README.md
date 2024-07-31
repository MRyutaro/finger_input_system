# finger_input_system

## 概要

![demo](docs/demo.gif)

カメラと両手のみを必要とする入力システムです。

カメラで手を認識し、その手の動きによって0から9までの数字を入力することができます。

[説明資料(PDF)](docs/explaination.pdf)

## 環境構築
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python hello.py`

`python hello.py`でエラーが出なければ環境構築完了です。もし黒い画面が出た場合はhello.py内のCAMERA_IDを変更してください。

## デモ
- `python main.py`
