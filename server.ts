import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Route for Model Inference
  app.post("/api/predict", (req, res) => {
    const { model, inputs } = req.body;
    const result = runSimulation(model, inputs);
    res.json(result);
  });

  // API Route for Batch Inference
  app.post("/api/predict-batch", (req, res) => {
    const { model, batch } = req.body;
    
    // Process top 1000 records to prevent timeout while showing capability
    const processed = batch.slice(0, 1000).map((inputs: any) => {
      return runSimulation(model, inputs);
    });

    const fraudCount = processed.filter((r: any) => r.type === 'fraud').length;

    res.json({
      summary: {
        total_processed: processed.length,
        fraud_found: fraudCount,
        legit_found: processed.length - fraudCount,
        fraud_rate: ((fraudCount / processed.length) * 100).toFixed(2) + '%'
      },
      results: processed.slice(0, 50), // Return sample for UI
      metadata: {
        timestamp: new Date().toISOString(),
        engine: "Batch-Aegis-v1"
      }
    });
  });

  function runSimulation(model: string, inputs: any) {
    let isFraud = false;
    let score = 0;

    const V17 = parseFloat(inputs.V17) || 0;
    const V14 = parseFloat(inputs.V14) || 0;
    const V12 = parseFloat(inputs.V12) || 0;
    const V10 = parseFloat(inputs.V10) || 0;
    const V11 = parseFloat(inputs.V11) || 0;
    const Amount = parseFloat(inputs.Amount) || 0;

    if (model === 'XGBoost + SMOTE') {
      const fraudScore = (V17 * 2.2) + (V14 * 1.6) + (V12 * 1.3) + (V10 * 1.1);
      isFraud = fraudScore < -4.8 || (V17 < -1.6 && Amount > 400);
      score = isFraud ? 85.4 + Math.random() * 8 : 98.8 - Math.random() * 4;
    } else if (model === 'Logistic Regression') {
      isFraud = (V17 + V14 + V12) < -3.8;
      score = isFraud ? 68.2 + Math.random() * 12 : 89.4 - Math.random() * 6;
    } else {
      isFraud = V17 < -2.4 || Amount > 2500 || (V14 < -2.2 && V11 < -1.8);
      score = isFraud ? 78.8 + Math.random() * 9 : 95.2 - Math.random() * 3;
    }

    return {
      type: isFraud ? 'fraud' : 'legit',
      score: parseFloat(score.toFixed(1)),
      id: Math.random().toString(36).substring(7).toUpperCase()
    };
  }

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`PRODUCTION SERVER deployed at http://localhost:${PORT}`);
  });
}

startServer();
