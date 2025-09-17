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

* **Detecção de faces**: OpenCV DNN (SSD/Caffe) com cache de modelo; fallback para HaarCascade.
* **Liveness**: combinação de sinais + limiar dinâmico (piora em ambientes ruins e **endurece** se houver indícios de spoof).
* **Reconhecimento**: InsightFace `buffalo_l` (w600k\_r50) via ONNXRuntime (providers=CPUExecutionProvider por padrão).
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
export INSIGHTFACE_HOME=/workspace/.insightface   # cache dos modelos do InsightFace
export TRAIN_DIR=/workspace/data/train             # onde salvamos dataset (live/spoof)

# Estabilidade/segurança de threads (evita segfaults em alguns ambientes)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Desliga warmup de modelos no startup (opcional)
export DISABLE_WARMUP=1

# Classificador ML opcional para spoof (se treinado)
export LIVENESS_CLF_PATH=/workspace/models/liveness_lr.joblib

# Parâmetros do liveness (pesos e limiares dinâmicos)
export LIVENESS_MIN_SIDE=224
export LIVENESS_W_BLUR=0.35   # pesos positivos
export LIVENESS_W_SAT=0.20
export LIVENESS_W_FREQ=0.45
export LIVENESS_W_AXIS=0.25   # penalizações (spoof)
export LIVENESS_W_GLARE=0.12
export LIVENESS_W_LINES=0.15
export LIVENESS_W_SHARP_SMALL=0.12
export LIVENESS_AXIS_SUSPECT=0.35
export LIVENESS_GLARE_SUSPECT=0.08
export LIVENESS_LINES_SUSPECT=0.10

# Liveness via vídeo (amostragem e votação)
export VIDEO_LIVENESS_MAX_FRAMES=24
export VIDEO_LIVENESS_MIN_FRAMES=8
export VIDEO_LIVENESS_PASS_RATIO=0.6
```

> **INSIGHTFACE\_HOME** evita re-download do `buffalo_l` a cada startup.

---

## ▶️ Execução

Para evitar problemas com http2/uvloop em alguns servidores, rode com HTTP/1.1 (`--http h11`):

```bash
uvicorn app.main:app \
  --host 0.0.0.0 --port 8081 \
  --loop asyncio --http h11 --log-level info
```

### Health check

```bash
curl -i http://localhost:8081/health
```

### Expor via Cloudflare Tunnel (opcional)

```bash
# Em outra janela / host
cloudflared tunnel --url http://localhost:8081
# Saída: https://<algo>.trycloudflare.com  ➜ use como BASE nas chamadas
```

---

## 📡 Endpoints

### `GET /health`

Retorna status, versão, Python e info básica de CUDA (sempre false no modo CPU).

---

### `POST /v1/liveness`

**Multipart** (`video` para clipes curtos ou `image` para fotos) ou **JSON** (`image_base64`). Parâmetros opcionais:

* `detector_backend` (padrão `opencv`)
* `threshold` (padrão `0.5` — pode ser dinamicamente ajustado)

```bash
# Multipart (vídeo)
curl -s -X POST "$BASE/v1/liveness?detector_backend=opencv&threshold=0.5" \
  -F "video=@/tmp/clip.webm;type=video/webm" | jq

# Multipart (fallback em foto)
curl -s -X POST "$BASE/v1/liveness?detector_backend=opencv&threshold=0.5" \
  -F "image=@$HOME/foto.png;type=image/png" | jq
```

**Resposta**

```json
{
  "faces": [
    {
      "is_live": true,
      "score": 0.707,
      "threshold": 0.50,
      "bbox": {"x":79,"y":367,"w":736,"h":736},
      "extra": {
        "detector_backend": "opencv-dnn",
        "explain": {
          "blur": 1.0,
          "saturation": 0.23,
          "freq_high_ratio": 0.69,
          "axis_anisotropy": 0.12,
          "glare": 0.01,
          "line_ratio": 0.02,
          "penalty_sharp_small": 0.00
        }
      }
    }
  ],
  "latency_ms": 500
}
```

> O **threshold mostrado é o efetivo**, após ajustes dinâmicos (ex.: sobe se detectar sinais de spoof; desce levemente em ambientes difíceis). Para vídeo, o retorno agrega múltiplos frames (ratio de “live” vs. “spoof”) e inclui estatísticas em `extra.per_frame`.

---

### `POST /v1/register`

Executa **liveness** (aceitando vídeo curto ou foto) e, se aprovado, extrai embedding (InsightFace) da melhor frame para salvar em `data/embeddings/{user_id}.npz`.

Parâmetros (query/form + multipart):

* `user_id` (obrigatório)
* `detector_backend` (opcional)
* `threshold` (opcional)

```bash
curl -i --max-time 60 \
  -F "video=@/tmp/clip.webm;type=video/webm" \
  "$BASE/v1/register?user_id=15&detector_backend=opencv&threshold=0.5"
```

**Resposta**

```json
{"success":true,"user_id":"15","liveness_ok":true,"message":"Cadastrado com sucesso"}
```

---

### `POST /v1/verify`

Executa **liveness** (clipe de vídeo recomendado) ➜ embedding ➜ compara c/ cadastro de `user_id`.

Parâmetros (query/form + multipart):

* `user_id` (obrigatório)
* `detector_backend`, `threshold` (opcionais)
* `match_threshold` (padrão **0.35**, distância cosseno; menor = mais estrito)

```bash
curl -i --max-time 60 \
  -F "video=@/tmp/clip.webm;type=video/webm" \
  "$BASE/v1/verify?user_id=15&detector_backend=opencv&threshold=0.40&match_threshold=0.35"
```

**Resposta** (exemplo)

```json
{
  "success": true,
  "user_id": "15",
  "liveness_ok": true,
  "match": true,
  "cosine_similarity": 0.942,
  "cosine_distance": 0.058,
  "match_threshold": 0.35,
  "message": "OK"
}
```

---

### `POST /v1/dataset/upload`

Salva imagem rotulada e recorte de face para treino.

Query/Form:

* `label`: **live** ou **spoof** (obrigatório)
* `user_id`: opcional (metadado)
* Aceita também `detector_backend` e `threshold` (query ou form) como nas rotas de liveness.

Estrutura de diretórios (padrão):

```
$TRAIN_DIR/
  live/
    raw/   # imagem original
    faces/ # crop da face
  spoof/
    raw/
    faces/
```

```bash
# LIVE
curl -s -X POST "$BASE/v1/dataset/upload?label=live&detector_backend=opencv&threshold=0.5" \
  -F "image=@$HOME/foto_real.png;type=image/png" | jq

# SPOOF
a
curl -s -X POST "$BASE/v1/dataset/upload?label=spoof&detector_backend=opencv&threshold=0.5" \
  -F "image=@$HOME/foto2.png;type=image/png" | jq
```

**Resposta** (exemplo)

```json
{
  "success": true,
  "label": "spoof",
  "paths": {"raw": "/workspace/data/train/spoof/raw/....jpg", "face": "/workspace/data/train/spoof/faces/....jpg"},
  "face": {"is_live": false, "score": 0.36, "threshold": 0.50, "bbox": {"x":...,"y":...,"w":...,"h":...}}
}
```

---

### `GET /v1/dataset/stats`

Resumo do dataset construído.

```bash
curl -s "$BASE/v1/dataset/stats" | jq
```

**Resposta**

```json
{ "total": 123, "live": 70, "spoof": 53, "samples": {"live": ["..."], "spoof": ["..."]} }
```

---

## 🧪 Treino de classificador anti-spoof (opcional)

Com as imagens de `TRAIN_DIR/live|spoof/faces`, é possível treinar um **Logistic Regression/SVM** leve em cima das features do liveness. Exemplo de script (fornecido em `scripts/train_liveness.py`):

```bash
python -m scripts.train_liveness --train_dir "$TRAIN_DIR" --out /workspace/models/liveness_lr.joblib
export LIVENESS_CLF_PATH=/workspace/models/liveness_lr.joblib
# reinicie o servidor; o liveness passará a usar o modelo como score principal
```

> Mesmo com o classificador, o sistema mantém **veto heurístico**: se sinais de spoof forem fortes (anisotropia/glare/linhas), o limiar efetivo **sobe**, tornando a decisão mais rígida.

---

## 📝 Logs

Formato:

```
2025-09-17 17:08:26,670 INFO api [req=abcd1234] [liveness-core] backend=opencv img=899x1600 thr=0.5
2025-09-17 17:08:26,671 INFO api [req=abcd1234] [liveness-core] face#0 score=0.707 live=True thr_eff=0.50 axis=0.12 glare=0.01 lines=0.02
```

* Defina `--log-level debug` para logs de multipart e detalhes.

---

## 🆘 Troubleshooting

* **Re-download de modelos InsightFace a cada start**: defina `INSIGHTFACE_HOME` para um caminho persistente (ex.: `/workspace/.insightface`).
* **Segmentation fault**: use `--http h11`, limite threads (`OMP_NUM_THREADS=1` etc.), e desabilite warmup (`DISABLE_WARMUP=1`). Confira também `libgl1` instalado.
* **Nenhum rosto detectado**: use imagens maiores/centradas; aumente `LIVENESS_MIN_SIDE` (ex.: 256); tente `detector_backend=opencv` (é o padrão).
* **Anti-spoof aprovando spoof específico**: suba pesos de penalização (`W_AXIS/GLARE/LINES/SHARP_SMALL`) ou treine um classificador com amostras reais do caso.
* **Verify retornando 404**: garanta que `/v1/register` foi executado para o `user_id` em questão.

---

## 📄 Licença

Uso interno / demo. Ajuste conforme a sua necessidade.

---

## 🙌 Créditos & Terceiros

* **OpenCV** (detecção DNN/HAAR)
* **InsightFace** (embeddings faciais, ONNX)
* **FastAPI / Pydantic**
* **ONNXRuntime**
