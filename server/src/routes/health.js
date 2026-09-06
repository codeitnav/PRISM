import { Router } from 'express'

const router = Router()

router.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok', service: 'server', timestamp: new Date().toISOString() })
})

export default router
