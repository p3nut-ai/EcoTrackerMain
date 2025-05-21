const puppeteer = require('puppeteer-extra');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

// Global temporary database array
let tempDataBase = [];

// TEMP DATABASE function
function add_temp_database(data) {
  if (!data) {
    tempDataBase = [];
  }
  
  // If the data object doesn’t have an "id", generate one.
  if (!data.id) {
    data.id = uuidv4();
  }

  if (!tempDataBase.find(existing => JSON.stringify(existing.data) === JSON.stringify(data.data))) {
    tempDataBase.push(data);
    console.log('Item added:', data.id);
    
    let endpoint = '';
    let payloadToSend = null;
    
    if (data.origin === "X") { // Twitter source
      endpoint = 'http://127.0.0.1:8000/api/twitter';
      payloadToSend = data.data;
    } else {
      // Fallback (if needed)
      endpoint = 'http://127.0.0.1:8000/api/data';
      payloadToSend = { data: data.data };
    }
    
    // Send the correctly structured payload to the API endpoint.
    axios.post(endpoint, payloadToSend)
      .then(response => {
        console.log('Data sent successfully:', response.data);
        // Clear the temporary database after successful sending.
        tempDataBase = [];
      })
      .catch(error => {
        console.error('Error sending data:', error.message);
      });
  } else {
    console.log('Duplicate item ignored:', data.id);
  }
}

// ------------------------------
// Twitter Scraper Function
// ------------------------------
async function scrapeUserTweets(page, keywords) {
  const searchUrl = 'https://x.com/DBNewswire';
  await page.goto(searchUrl, { waitUntil: 'networkidle2' });
  await page.waitForSelector('article', { timeout: 30000 });

  const today = new Date();
  const currentDate = today.toISOString().split('T')[0];
  // const currentDate = "2025-03-07";  // You can update this as needed for debugging shit lang

  console.log("Current Date:", currentDate);

  // Evaluate tweets on the page and filter based on current date and keywords
  const tweets = await page.evaluate((currentDate, keywords) => {
    const tweetNodes = document.querySelectorAll('article');
    const tweetData = [];
    tweetNodes.forEach(node => {
      const timeElement = node.querySelector('time');
      const tweetTextElement = node.querySelector('div[data-testid="tweetText"]');
      
      if (timeElement && tweetTextElement) {
        const tweetDate = timeElement.getAttribute('datetime')?.split('T')[0];
        const tweetText = tweetTextElement.textContent;
        // Check if the tweet date matches today's date
        if (tweetDate === currentDate &&
            keywords.some(keyword => tweetText.includes(keyword))) {
          tweetData.push(tweetText);
        }
      } else {
        console.log("Error: Unable to locate tweet elements");
      }
    });
    return tweetData;
  }, currentDate, keywords);

  if (tweets.error) {
    console.error("Error from evaluate:", tweets.error);
  }

  // Post the Twitter data (payload with origin "X" and data array of tweet texts)
  add_temp_database({ origin: "X", data: tweets });
  
  return tweets;
}

// ------------------------------
// Function to Auto-Refresh and Re-Run the Scraper Every 5 Minutes
// ------------------------------
function startAutoRefreshWithScrape(page, pageName, scraperFunction, ...scraperArgs) {
  const REFRESH_INTERVAL = 1 * 60 * 1000; 
  setInterval(async () => {
    try {
      await page.reload({ waitUntil: 'networkidle2' });
      console.log(`${pageName} refreshed at ${new Date().toLocaleTimeString()}`);
      
      const result = await scraperFunction(page, ...scraperArgs);
      console.log(`Updated ${pageName} scraping result:`, result);
    } catch (error) {
      console.error(`Error refreshing ${pageName}:`, error);
    }
  }, REFRESH_INTERVAL);
}

// ------------------------------
// Main Function: Set Up Page and Start Continuous Scraping
// ------------------------------
(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", // Adjust path if needed
    headless: false,
  });
  
  const twitterPage = await browser.newPage();
  
  const userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36';
  await twitterPage.setUserAgent(userAgent);
  await twitterPage.setViewport({ width: 1280, height: 800 });
  
  await twitterPage.goto('https://x.com/login', { waitUntil: 'networkidle2' });
  await twitterPage.waitForSelector('article', { timeout: 0 });
  
  const keywords = ["dollar", "Federal Reserve", "interest rate", "monetary policy", "US economy", "S&P", "USD", "US", "TRUMP"];
  
  const initialTweets = await scrapeUserTweets(twitterPage, keywords);
  console.log("Initial Filtered Tweets:", initialTweets);
  
  startAutoRefreshWithScrape(twitterPage, "Twitter Page", scrapeUserTweets, keywords);
})();
