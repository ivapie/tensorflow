import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import temperatures from './temperatures.json'
import temperaturesTesting from './temperatures.json'

const trainingData = tf.tensor2d(temperatures.map( item => [
  item.temperatureC
]))
const outputData = tf.tensor2d(temperatures.map(item => [
  item.value === "Baja" ? 1 : 0,
  item.value === "Normal" ? 1 : 0,
  item.value === "Alta" ? 1 : 0,
  item.value === "Muy Alta" ? 1 : 0,
  item.value === "Extremadamente Alta" ? 1 : 0,
]))
const testingData = tf.tensor2d(temperaturesTesting.map( item => [
  item.temperatureC
]))

const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [1],
  activation: "sigmoid",
  units: 5,
}))

model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 5,
}))

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})

// train/fit our network
const startTime = Date.now()
model.fit(trainingData, outputData, {epochs: 100})
  .then((history) => {
    // console.log(history)
    model.predict(testingData).print( )
  })