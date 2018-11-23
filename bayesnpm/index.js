
var bayes = require('bayes');
var classifier = bayes();
var spamTweets = require("/Users/delphineshi/Downloads/temp/tweets_spam.json");
var infoTweets = require("/Users/delphineshi/Downloads/temp/tweets_info.json");

const data = [{name: 'info', tweets: infoTweets}, {name: 'spam', tweets: spamTweets}];

// for (var category of data) {
//   console.log(`training model with ${category.name} data.`)
//   for (var tweet of category.tweets) {
//     classifier.learn(tweet.text, category.name);
//   }
// }
//
// console.log(classifier.categorize("Brexit : La Grande-Bretagne, grande perdante")); //info
// console.log(classifier.categorize("Top cool j'adore !")); //spam
// console.log(classifier.categorize("Supportons les gilets jaunes, la hausse des carburants est intolérable")); //info

var totalDataCount = spamTweets.length + infoTweets.length;
var tp = 0;
var tn = 0;
var fp = 0;
var fn = 0;

var t0 = new Date().getTime();

// Iterate through every data element index
for (var testIndex=0; testIndex < totalDataCount; testIndex++){
  console.log(testIndex)
  // instantiate a new model
  var classifier = bayes();
  var testData = [];
  var counter = 0;
  for (var category of data) {
    for (var tweet of category.tweets) {
      counter ++;
      if (counter === testIndex) {
        // If equal to test Index then ommit from training.
        testData.push({category: category.name, tweet: tweet});
      } else {
        // Train on all other data elements.
        classifier.learn(tweet.text, category.name);
      }
    }
  }
  // Use test data.
  for (var test of testData) {
    if (classifier.categorize(test.tweet.text) === test.category) {
      if (test.category === 'info') {
        tp++;
      } else {
        tn ++;
      }
    } else {
      if (test.category === 'info') {
        fp++;
      } else {
        fn++;
      }
    }
  }
}
var t1 = new Date().getTime();

console.log('total tests: ', (tp + tn + fp + fn));
console.log(`TP = ${tp}`);
console.log(`TN = ${tn}`);
console.log(`FP = ${fp}`);
console.log(`FN = ${fn}`);
console.log('Took ' + (t1 - t0) + ' milliseconds.')
accuracy = (tp + tn) / (tp + tn + fp + fn)
console.log(`Accuracy = ${accuracy}`)
