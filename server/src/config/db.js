import mongoose from 'mongoose'
import { env } from './env.js'

export async function connectMongo() {
  await mongoose.connect(env.mongoUri)
  console.log(`[server] connected to MongoDB at ${env.mongoUri}`)
}
