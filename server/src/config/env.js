import 'dotenv/config'

export const env = {
  port: process.env.PORT || 4000,
  mongoUri: process.env.MONGO_URI || 'mongodb://mongo:27017/prism',
  mlServiceUrl: process.env.ML_SERVICE_URL || 'http://ml:8000',
  nodeEnv: process.env.NODE_ENV || 'development',
}
