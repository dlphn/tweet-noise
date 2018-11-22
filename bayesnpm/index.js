
var bayes = require('bayes');
var classifier = bayes();
var spamTweets = require("/Users/delphineshi/Downloads/temp/tweets_spam.json");
var infoTweets = require("/Users/delphineshi/Downloads/temp/tweets_info.json");

const data = [{name: 'info', tweets: infoTweets}, {name: 'spam', tweets: spamTweets}];

for (var category of data) {
  console.log(`training model with ${category.name} data.`)
  for (var tweet of category.tweets) {
    classifier.learn(tweet.text, category.name);
  }
}


console.log(classifier.categorize("Brexit : La Grande-Bretagne, grande perdante")); //info
console.log(classifier.categorize("Top cool j'adore !")); //spam
console.log(classifier.categorize("Supportons les gilets jaunes, la hausse des carburants est intolérable")); //info
