# Anti-Spoofing & Face Recognition API

API leve em **FastAPI** para **liveness (anti-spoofing)** e **reconhecimento facial** (registro/validação) 100% em **CPU**, com detecção via **OpenCV DNN**, embeddings via **InsightFace (ONNXRuntime)** e rotas para montar **dataset** e treinar um classificador simples de spoof.

---

## ✨ Recursos

* **/v1/liveness**: heurísticas + sinais anti-spoof (blur, alta-freq, anisotropia de grade, glare/reflexo, linhas retas, etc.) com **limiar dinâmico**, agora aceitando **imagem única ou clipe curto de vídeo**.
* **/v1/register**: liveness ➜ embedding (InsightFace) ➜ salva em `data/embeddings/{user_id}.npz`.
* **/v1/verify**: liveness ➜ embedding ➜ compara com cadastro (cosseno), com `match_threshold`.
* **/v1/dataset/upload**: salva imagens rotuladas (`live`/`spoof`) e crops de face para treino.
* **/v1/dataset/stats**: contagem e exemplos do dataset.
* **CPU-first**: roda sem GPU. Se houver GPU compatível, ONNXRuntime pode usar CUDA (opcional).
* **Logs claros** com `req-id` por chamada.

---

## 🧱 Arquitetura (resumo)

* **Detecção de faces**: suporta **YuNet (ONNX)**, **OpenCV DNN (SSD/Caffe)** quando arquivos já estão presentes e fallback para **HaarCascade**, sem download em tempo de requisição.
* **Liveness**: combinação de sinais + limiar dinâmico (piora em ambientes ruins e **endurece** se houver indícios de spoof).
* **Reconhecimento**: InsightFace `buffalo_l` (w600k_r50) via ONNXRuntime (providers=CPUExecutionProvider por padrão).
* **Dataset**: imagens salvas em `TRAIN_DIR/live|spoof/{raw,faces}`.

---

## ✅ Requisitos

* **Python 3.10+** (recomendado 3.12)
* Sistema com **libGL** para OpenCV (em Ubuntu: `apt-get install -y libgl1`)

### Dependências principais

```
fastapi
uvicorn
numpy==1.26.4
opencv-python==4.10.0.84
pydantic==2.9.2
python-multipart
python-dotenv
insightface
onnxruntime
joblib
scikit-learn
Pillow==10.4.0
```

> Observação: **TensorFlow/PyTorch não são necessários** para o caminho padrão (CPU + ONNXRuntime). Se tiver instalado TF em GPU muito nova, forçamos CPU para evitar erros.

---

## 🚀 Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se faltar `libgl` no host (para OpenCV):

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y libgl1
```

---

## 🔧 Configuração

Variáveis de ambiente úteis:

```bash
# Diretórios
export INSIGHTFACE_HOME=/workspace/.insightface
export TRAIN_DIR=/workspace/data/train
export OPENCV_FACE_DIR=/workspace/models/opencv_face

# Estabilidade/segurança de threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Desliga warmup
export DISABLE_WARMUP=1

# Detectores
export OPENCV_DNN_DISABLE=0
export FORCE_OPENCV_DETECTOR=0
export YUNET_SCORE_THRESHOLD=0.6
export YUNET_NMS_THRESHOLD=0.3

# Classificador spoof
export LIVENESS_CLF_PATH=/workspace/models/liveness_lr.joblib

# Pesos
export LIVENESS_MIN_SIDE=224
export LIVENESS_W_BLUR=0.35
export LIVENESS_W_SAT=0.20
export LIVENESS_W_FREQ=0.45
export LIVENESS_W_AXIS=0.25
export LIVENESS_W_GLARE=0.12
export LIVENESS_W_LINES=0.15
export LIVENESS_W_SHARP_SMALL=0.12
export LIVENESS_AXIS_SUSPECT=0.35
export LIVENESS_GLARE_SUSPECT=0.08
export LIVENESS_LINES_SUSPECT=0.10

# Vídeo
export VIDEO_LIVENESS_MAX_FRAMES=24
export VIDEO_LIVENESS_MIN_FRAMES=8
export VIDEO_LIVENESS_PASS_RATIO=0.6
```

---

## ▶️ Execução

```bash
uvicorn app.main:app \
  --host 0.0.0.0 --port 8081 \
  --loop asyncio --http h11 --log-level info
```

### Health check

```bash
curl -i http://localhost:8081/health
```

---

## 📡 Endpoints

### `GET /health`

Retorna status, versão, Python e info básica de CUDA.

---

### `POST /v1/liveness`

```bash
# Vídeo (YuNet)
curl -s -X POST "$BASE/v1/liveness?detector_backend=yunet&threshold=0.5" \
  -F "video=@/tmp/clip.webm;type=video/webm" | jq

# Foto (OpenCV)
curl -s -X POST "$BASE/v1/liveness?detector_backend=opencv&threshold=0.5" \
  -F "image=@$HOME/foto.png;type=image/png" | jq
```

---

### `POST /v1/register`

```bash
curl -i --max-time 60 \
  -F "video=@/tmp/clip.webm;type=video/webm" \
  "$BASE/v1/register?user_id=15&detector_backend=opencv&threshold=0.5"
```

---

### `POST /v1/verify`

```bash
curl -i --max-time 60 \
  -F "video=@/tmp/clip.webm;type=video/webm" \
  "$BASE/v1/verify?user_id=15&detector_backend=opencv&threshold=0.40&match_threshold=0.35"
```

---

### `POST /v1/dataset/upload`

```bash
# LIVE
curl -s -X POST "$BASE/v1/dataset/upload?label=live&detector_backend=yunet&threshold=0.5" \
  -F "image=@$HOME/foto_real.png;type=image/png" | jq

# SPOOF
curl -s -X POST "$BASE/v1/dataset/upload?label=spoof&detector_backend=haar&threshold=0.5" \
  -F "image=@$HOME/foto2.png;type=image/png" | jq
```

---

### `GET /v1/dataset/stats`

```bash
curl -s "$BASE/v1/dataset/stats" | jq
```

---

## 🧪 Treino de classificador

```bash
python -m scripts.train_liveness --train_dir "$TRAIN_DIR" --out /workspace/models/liveness_lr.joblib
export LIVENESS_CLF_PATH=/workspace/models/liveness_lr.joblib
```

---

## 📝 Logs

```
2025-09-17 17:08:26,670 INFO api [req=abcd1234] [liveness-core] backend=opencv img=899x1600 thr=0.5
```

---

## 🆘 Troubleshooting

* Defina `INSIGHTFACE_HOME` para evitar re-download.
* Use `--http h11` se ocorrer segmentation fault.
* Ajuste `W_AXIS/GLARE/LINES` se spoof passar.
* Use `detector_backend=yunet` para melhorar detecção.

---

## 📄 Licença

Uso interno / demo.

---

## 🙌 Créditos

* **OpenCV**
* **InsightFace**
* **FastAPI / Pydantic**
* **ONNXRuntime**