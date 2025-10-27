const map = { Z:'P', E:'O', N:'L', I:'A', T:'R', P:'Z', O:'E', L:'N', A:'I', R:'T' };
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

// Dataset
const pairs = alphabet.map(letter => ({ x: letter, y: map[letter] || letter }));

// One-hot
const letterToIndex = {}, indexToLetter = {};
alphabet.forEach((l,i)=>{ letterToIndex[l]=i; indexToLetter[i]=l; });
function toOneHot(index){ const arr=Array(alphabet.length).fill(0); arr[index]=1; return arr; }
const xs = tf.tensor2d(pairs.map(p=>toOneHot(letterToIndex[p.x])));
const ys = tf.tensor2d(pairs.map(p=>toOneHot(letterToIndex[p.y])));

// Modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 64, inputShape:[26], activation:'relu' }));
model.add(tf.layers.dense({ units:26, activation:'softmax' }));
model.compile({ optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy'] });

// Treino
async function trainModel(){
  document.getElementById("status").innerText="Treinando modelo...";
  await model.fit(xs, ys, {
    epochs:200,
    callbacks:{
      onEpochEnd:(epoch, logs)=>{
        document.getElementById("status").innerText=
          `Época ${epoch+1}/200 — perda:${logs.loss.toFixed(5)} — acurácia:${(logs.acc*100).toFixed(2)}%`;
      },
      onTrainEnd:()=>{
        document.getElementById("status").innerText="✅ Treino concluído!";
        document.getElementById("predictBtn").disabled=false;
      }
    }
  });
}

// Previsão
async function predictText(){
  const text=document.getElementById("inputText").value.toUpperCase();
  let result='';
  for(let ch of text){
    if(!/[A-Z]/.test(ch)){ result+=ch; continue; }
    const input=tf.tensor2d([toOneHot(letterToIndex[ch])]);
    const output=model.predict(input);
    const data=await output.data();
    const predictedIndex=data.indexOf(Math.max(...data));
    result+=indexToLetter[predictedIndex];
  }
  document.getElementById("output").innerText=result;
}

document.getElementById("trainBtn").addEventListener("click",trainModel);
document.getElementById("predictBtn").addEventListener("click",predictText);