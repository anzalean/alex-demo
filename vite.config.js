import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        mp: resolve(__dirname, 'page-opencv-mp.html'),
        tf: resolve(__dirname, 'page-opencv-tf.html'),
        webgazer: resolve(__dirname, 'page-opencv-webgazer.html')
      }
    }
  }
})