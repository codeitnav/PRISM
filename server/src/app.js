import express from 'express'
import cors from 'cors'
import morgan from 'morgan'
import healthRouter from './routes/health.js'

export function createApp() {
  const app = express()

  app.use(cors())
  app.use(morgan('dev'))
  app.use(express.json())

  app.use('/', healthRouter)

  // TODO(Task 0.2): mount /api/reconstruct once the shared API contract is finalized
  // TODO(Task 3.1): wire reconstruction orchestration route -> ML service -> MongoDB

  app.use((req, res) => {
    res.status(404).json({ error: 'not_found', path: req.path })
  })

  return app
}
