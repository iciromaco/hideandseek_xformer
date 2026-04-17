# XML設定リファインツール

XMLの物理設定を手動でリファインするためのシンプルなツールです。`EnvXMLBuilder` から不要な機能を削ぎ落とし、**XMLビルド機能**と**ヒューマンモード（手動操作）**のみを抽出しました。

## 特徴

- ✅ **最小化されたXML生成**: アリーナ、エージェント、オブジェクトの定義のみ
- ✅ **ヒューマンコントロール**: キーボード操作でエージェントを直感的に制御
- ✅ **CLI対応**: コマンドライン引数で柔軟に設定可能
- ✅ **XML保存機能**: リファイン後のXML設定を簡単に保存
- ✅ **スタンドアロン**: 外部依存を最小化（MuJoCoと`pynput`のみ）

## 必要な依存パッケージ

```bash
pip install mujoco pynput
```

## 使用方法

### 1. ビューアで対話的に実行

```bash
python xml_refinement_tool.py
```

**キーボード操作**:
- `w`: 前進
- `s`: 後退  
- `a`: 左回転
- `d`: 右回転
- `Ctrl+C`: 終了

### 2. XMLをファイルに保存

```bash
python xml_refinement_tool.py --save-xml config.xml
```

### 3. XMLをコンソール出力

```bash
python xml_refinement_tool.py --print-xml
```

### 4. パラメータを調整

```bash
python xml_refinement_tool.py \
  --n-seekers 1 \
  --n-hiders 2 \
  --n-boxes 3 \
  --n-ramps 2 \
  --duration 120 \
  --save-xml refined_config.xml
```

## 利用可能なオプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--n-seekers` | 1 | シーカーの数 |
| `--n-hiders` | 2 | ハイダーの数 |
| `--n-boxes` | 2 | 箱のオブジェクト数 |
| `--n-ramps` | 1 | ランプのオブジェクト数 |
| `--duration` | 60.0 | 実行時間（秒） |
| `--learnable-agent` | s | 制御対象のエージェント（例: `s`, `h1`） |
| `--no-interactive` | - | キーボード入力を無効化 |
| `--save-xml` | - | XML保存先ファイル |
| `--print-xml` | - | XMLをコンソール出力 |

## Pythonスクリプトからの使用

```python
from xml_refinement_tool import XMLRefinementApp, MinimalEnvConfig, XMLBuilder

# 方法1: ビューア付きで実行
app = XMLRefinementApp(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1)
app.run(duration=60)

# 方法2: XMLのみを取得
app = XMLRefinementApp(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, interactive=False)
xml_string = app.get_xml_string()
with open("output.xml", "w") as f:
    f.write(xml_string)

# 方法3: MuJoCoモデルを直接取得
import mujoco
config = MinimalEnvConfig(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1)
builder = XMLBuilder(config)
xml_str = builder.build_xml()
model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)
```

## XML物理設定パラメータの調整

`MinimalEnvConfig` クラスで以下の物理設定を直接編集できます:

### アリーナ設定
```python
ARENA_HALF = 6.0  # アリーナサイズ（半幅）
```

### エージェント設定
```python
AGENT_MASS = 1.0                    # エージェント質量
AGENT_Z_POS = 0.1                  # エージェント初期高さ
AGENT_Z_MIN = 0.0                  # 最低高さ制限
AGENT_Z_MAX = 1.0                  # 最高高さ制限
AGENT_DAMPING_XY = 0.0             # XY平面のダンピング
AGENT_DAMPING_Z = 0.0              # 垂直ダンピング
AGENT_DAMPING_ROT = 0.0            # 回転ダンピング
AGENT_ACTUATOR_FWD = "1200 -100 -10"  # 前進アクチュエータ（gainprm）
AGENT_TURN_GAIN = "120 -100 -10"       # ターンゲイン
```

### オブジェクト設定
```python
BOX_MASS = 0.5                      # 箱の質量
BOX_JOINT_DAMPING = 0.001          # 箱の関節ダンピング
RAMP_MASS = 0.5                    # ランプ質量
RAMP_JOINT_DAMPING = 0.001         # ランプ関節ダンピング
RAMP_INNER_WEIGHT_MASS = 0.1       # ランプ内部ウェイト質量
```

## ワークフロー例

### 1. XMLビルド + 手動テスト
```bash
# ビューアで物理動作を確認
python xml_refinement_tool.py --duration 120
```

### 2. パラメータ調整 + 保存
```python
# スクリプト内でパラメータを微調整
from xml_refinement_tool import MinimalEnvConfig, XMLBuilder

class CustomConfig(MinimalEnvConfig):
    AGENT_MASS = 2.0  # 質量を2倍に
    RAMP_INNER_WEIGHT_MASS = 0.2  # ウェイトを増加

config = CustomConfig()
builder = XMLBuilder(config)

# XMLを保存
with open("custom_physics.xml", "w") as f:
    f.write(builder.build_xml())
```

### 3. 本メインスクリプトへの統合
```python
# main28_train_final.py などで使用
from xml_refinement_tool import XMLBuilder, MinimalEnvConfig

config = MinimalEnvConfig(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1)
builder = XMLBuilder(config)
xml_string = builder.build_xml()

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
```

## トラブルシューティング

### 「pynputがインストールされていない」エラー
```bash
pip install pynput
```

### ビューアが起動しない
```bash
# GUIなしで実行
python xml_refinement_tool.py --print-xml > output.xml
```

### MuJoCoエラー
```bash
pip install --upgrade mujoco
```

## ファイル構成

- `xml_refinement_tool.py`: メイン実行ツール
  - `MinimalEnvConfig`: 環境設定（最小構成）
  - `XMLBuilder`: XML生成
  - `HumanController`: キーボード入力処理
  - `XMLRefinementApp`: 統合主アプリケーション

---

**用途**: XML物理設定の手動リファインと動作確認
**対象者**: エンジニア向け開発ツール
