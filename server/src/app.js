import express from 'express'
import cors from 'cors'
import morgan from 'morgan'
import healthRouter from './routes/health.js'
import reconstructRouter from './routes/reconstruct.js'

export function createApp() {
  const app = express()

  app.use(cors())
  app.use(morgan('dev'))
  app.use(express.json())

  app.use('/', healthRouter)
  app.use('/', reconstructRouter)

  app.use((req, res) => {
    res.status(404).json({ error: 'not_found', path: req.path })
  })

  return app
}
