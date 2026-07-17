# 视频广告分析服务（R2 + Railway）

从 Cloudflare R2 拉取镜头切片 → 调用分析流水线 → 产物写回 R2 的 HTTP 服务。

## 接口
- `GET /health` 健康检查
- `POST /analyze` 触发一次分析
  ```json
  {"project_id":"test","product":"章鱼","pipeline":"merged","pro":false,"overwrite":false,"cross_compare":false}
  ```
  可选请求头 `X-API-Token`（设了 APP_TOKEN 才校验）

## 环境变量（Railway Variables / 本地 .env）
R2_ACCOUNT_ID、R2_ACCESS_KEY_ID、R2_SECRET_ACCESS_KEY、R2_BUCKET_NAME、
R2_ENDPOINT、R2_PUBLIC_URL、OPENROUTER_API_KEY、APP_TOKEN(可选)

## 本地起服务
```
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## 部署（Railway，连 GitHub 自动部署）
1. 推本仓库到 GitHub
2. Railway → New Project → Deploy from GitHub repo → 选本仓库
3. Variables 里填上面的环境变量
4. Settings → Networking → Generate Domain
5. `curl https://<domain>/health` 验证

构建用 Nixpacks，`nixpacks.toml` 已声明安装 ffmpeg。
