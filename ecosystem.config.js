const fs = require("fs");
const path = require("path");

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return;
  }

  for (const line of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      continue;
    }
    const [key, ...valueParts] = trimmed.split("=");
    if (!process.env[key]) {
      process.env[key] = valueParts.join("=").replace(/^['"]|['"]$/g, "");
    }
  }
}

loadEnvFile(path.join(__dirname, ".env"));

module.exports = {
  apps: [
    {
      name: "poker44-miner",
      cwd: "/home/poker44",
      script: "neurons/miner.py",
      interpreter: "/home/poker44/venv/bin/python",
      args: [
        "--netuid", process.env.NETUID || "126",
        "--wallet.name", process.env.WALLET_NAME || "pierre",
        "--wallet.hotkey", process.env.HOTKEY || "pierre1hotkey",
        "--subtensor.network", process.env.SUBTENSOR_NETWORK || "finney",
        "--axon.port", process.env.AXON_PORT || "7091",
      ].concat(
        process.env.AXON_EXTERNAL_IP
          ? ["--axon.external_ip", process.env.AXON_EXTERNAL_IP]
          : []
      ).concat(
        process.env.AXON_EXTERNAL_PORT
          ? ["--axon.external_port", process.env.AXON_EXTERNAL_PORT]
          : []
      ),
      env: Object.assign({}, process.env, {
        PYTHONUNBUFFERED: "1",
      }),
      out_file: "logs/poker44-miner.out.log",
      error_file: "logs/poker44-miner.err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 20,
      min_uptime: "30s",
    },
    {
      name: "poker44-miner2",
      cwd: "/home/poker44",
      script: "neurons/miner.py",
      interpreter: "/home/poker44/venv/bin/python",
      args: [
        "--netuid", "126",
        "--wallet.name", "pierre",
        "--wallet.hotkey", "pierrehotkey",
        "--subtensor.network", "finney",
        "--axon.port", "7092",
        "--axon.external_ip", "65.108.142.25",
        "--axon.external_port", "7092",
      ],
      env: Object.assign({}, process.env, {
        PYTHONUNBUFFERED: "1",
        AXON_PORT: "7092",
        AXON_EXTERNAL_PORT: "7092",
        HOTKEY: "pierrehotkey",
        POKER44_LOG_DIR: "logs/miner2",
      }),
      out_file: "logs/poker44-miner2.out.log",
      error_file: "logs/poker44-miner2.err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 20,
      min_uptime: "30s",
    },
  ],
};
