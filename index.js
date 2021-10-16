var net
const urls = ['https://i.imgur.com/HBrB8p0', 'https://imgur.com/UXN9GD8', 'https://imgur.com/e1FM4b8']
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
function save(model) {
  model.save()
}
async function app() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');
  
    // Create an object from Tensorflow.js data API which could capture image 
    // from the web camera as Tensor.
  
    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
      // Capture an image from the web camera.
      const img = changeImg()
  
      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(img, true);
  
      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);
  
      // Dispose the tensor to release the memory.
      img.dispose();
    };
    // When clicking a button, add an example for that class.
    document.getElementById('m1').addEventListener('click', () => addExample('m1'));
    document.getElementById('m2').addEventListener('click', () => addExample('m2'));
    document.getElementById('m3').addEventListener('click', () => addExample('m3'));
    document.getElementById('m4').addEventListener('click', () => addExample('m4'));
    document.getElementById('m5').addEventListener('click', () => addExample('m5'));
    document.getElementById('m6').addEventListener('click', () => addExample('m6'));
    document.getElementById('m7').addEventListener('click', () => addExample('m7'));
    document.getElementById('m8').addEventListener('click', () => addExample('m8'));
    document.getElementById('m9').addEventListener('click', () => addExample('m9'));
    document.getElementById('m10').addEventListener('click', () => addExample('m10'));
    document.getElementById('f1').addEventListener('click', () => addExample('f1'));
    document.getElementById('f2').addEventListener('click', () => addExample('f2'));
    document.getElementById('f3').addEventListener('click', () => addExample('f3'));
    document.getElementById('f4').addEventListener('click', () => addExample('f4'));
    document.getElementById('f5').addEventListener('click', () => addExample('f5'));
    document.getElementById('f6').addEventListener('click', () => addExample('f6'));
    document.getElementById('f7').addEventListener('click', () => addExample('f7'));
    document.getElementById('f8').addEventListener('click', () => addExample('f8'));
    document.getElementById('f9').addEventListener('click', () => addExample('f9'));
    document.getElementById('f10').addEventListener('click', () => addExample('f10'));
    document.getElementById('brady').addEventListener('click', () => addExample('s11'));
    document.getElementById('save').addEventListener('click', async () => {await net.save()});
    while (true) {
      if (classifier.getNumClasses() > 0) {
        const img = changeImg();
  
        // Get the activation from mobilenet from the webcam.
        const activation = net.infer(img, 'conv_preds');
        // Get the most likely class and confidence from the classifier module.
        const result = await classifier.predictClass(activation);
  
        const classes = ['A', 'B', 'C'];
        document.getElementById('console').innerText = `
          prediction: ${classes[result.label]}\n
          probability: ${result.confidences[result.label]}
        `;
  
        // Dispose the tensor to release the memory.
        img.dispose();
      }
  
      await tf.nextFrame();
    }
  }
  function changeImg() {
    const url = urls[Math.floor(Math.random() * urls.length)]
    document.getElementById("img").src=url;
    return document.getElementById("img")
  }
  app();
