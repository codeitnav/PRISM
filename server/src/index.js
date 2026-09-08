import { createApp } from './app.js'
import { connectMongo } from './config/db.js'
import { env } from './config/env.js'

await connectMongo()

const app = createApp()

app.listen(env.port, () => {
  console.log(`[server] listening on port ${env.port} (env: ${env.nodeEnv})`)
})
