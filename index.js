const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  // When clicking a button, add an example for that class.
  document.getElementById('m1').addEventListener('click', () => addExample(0));
  document.getElementById('m2').addEventListener('click', () => addExample(1));
  document.getElementById('m3').addEventListener('click', () => addExample(2));
  document.getElementById('m4').addEventListener('click', () => addExample(3));
  document.getElementById('m5').addEventListener('click', () => addExample(4));
  document.getElementById('m6').addEventListener('click', () => addExample(5));
  document.getElementById('m7').addEventListener('click', () => addExample(6));
  document.getElementById('m8').addEventListener('click', () => addExample(7));
  document.getElementById('m9').addEventListener('click', () => addExample(8));
  document.getElementById('m10').addEventListener('click', () => addExample(9));
  document.getElementById('f1').addEventListener('click', () => addExample(10));
  document.getElementById('f2').addEventListener('click', () => addExample(11));
  document.getElementById('f3').addEventListener('click', () => addExample(12));
  document.getElementById('f4').addEventListener('click', () => addExample(13));
  document.getElementById('f5').addEventListener('click', () => addExample(14));
  document.getElementById('f6').addEventListener('click', () => addExample(15));
  document.getElementById('f7').addEventListener('click', () => addExample(16));
  document.getElementById('f8').addEventListener('click', () => addExample(17));
  document.getElementById('f9').addEventListener('click', () => addExample(18));
  document.getElementById('f10').addEventListener('click', () => addExample(19));
  document.getElementById('dawg').addEventListener('click', () => changeImg());
  document.getElementById('brady').addEventListener('click', () => addExample(20));
  document.getElementById('save').addEventListener('click', async () => {await net.save()});

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 's11'];;
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

/*
const classes = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 's11'];
    document.getElementById('m1').addEventListener('click', () => addExample(0));
    document.getElementById('m2').addEventListener('click', () => addExample(1));
    document.getElementById('m3').addEventListener('click', () => addExample(2));
    document.getElementById('m4').addEventListener('click', () => addExample(3));
    document.getElementById('m5').addEventListener('click', () => addExample(4));
    document.getElementById('m6').addEventListener('click', () => addExample(5));
    document.getElementById('m7').addEventListener('click', () => addExample(6));
    document.getElementById('m8').addEventListener('click', () => addExample(7));
    document.getElementById('m9').addEventListener('click', () => addExample(8));
    document.getElementById('m10').addEventListener('click', () => addExample(9));
    document.getElementById('f1').addEventListener('click', () => addExample(10));
    document.getElementById('f2').addEventListener('click', () => addExample(11));
    document.getElementById('f3').addEventListener('click', () => addExample(12));
    document.getElementById('f4').addEventListener('click', () => addExample(13));
    document.getElementById('f5').addEventListener('click', () => addExample(14));
    document.getElementById('f6').addEventListener('click', () => addExample(15));
    document.getElementById('f7').addEventListener('click', () => addExample(16));
    document.getElementById('f8').addEventListener('click', () => addExample(17));
    document.getElementById('f9').addEventListener('click', () => addExample(18));
    document.getElementById('f10').addEventListener('click', () => addExample(19));
    document.getElementById('dawg').addEventListener('click', () => changeImg());
    document.getElementById('brady').addEventListener('click', () => addExample(20));
    document.getElementById('save').addEventListener('click', async () => {await net.save()});
*/