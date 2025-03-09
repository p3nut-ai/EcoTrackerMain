const puppeteer = require('puppeteer-extra');
const axios =  require('axios');
const {v4: uuidv4} = require('uuid');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());


// Global temporary database array
let tempDataBase = [];

// TEMP DATABASE function
function add_temp_database(data) {
  // Reset the temporary database if data is falsy
  if (!data) {
    tempDataBase = [];
  }
  
  // If the data object doesn’t have an "id", generate one.
  if (!data.id) {
    data.id = uuidv4();
  }

  // Check for duplicates by comparing the "data" property (deep comparison)
  if (!tempDataBase.find(existing => JSON.stringify(existing.data) === JSON.stringify(data.data))) {
    tempDataBase.push(data);
    console.log('Item added:', data.id);
    
    let endpoint = '';
    let payloadToSend = null;
    
    // Determine endpoint and payload based on the source (origin)
    if (data.origin === "FOREX") {
      endpoint = 'http://127.0.0.1:8000/api/forex';
      // Forex endpoint expects an object with a key "data" holding an array of objects.
      payloadToSend = { data: data.data };
    } else if (data.origin === "X") { // Twitter source
      endpoint = 'http://127.0.0.1:8000/api/twitter';
      // Twitter endpoint expects a raw array of strings.
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
// Forex Factory Scraper Function
// ------------------------------
const forexFactoryScraper = async (page) => {
  await page.goto('https://www.forexfactory.com/news', { waitUntil: 'networkidle2' });
  await page.waitForSelector('[data-compid="NewsRightThree"]', { timeout: 30000 });
  
  const moreButtonSelector = '[data-compid="NewsRightThree"] .flexMore';
  const isMoreButtonVisible = await page.$(moreButtonSelector);
  
  if (isMoreButtonVisible) {
 
    try {

    
      // Click the button.
      await page.click(moreButtonSelector);
      console.log("Button is pressed");
    } catch (error) {
      console.log('No "flexMore" button found or it is not clickable.', error);
    }
  } else {
    console.log('No "flexMore" button found.');
  }

  
  const forexFactoryNews = await page.evaluate(() => {
    const container = document.querySelector('[data-compid="NewsRightThree"]');
    if (!container) return [];
    
    const items = Array.from(container.querySelectorAll('li'));
    
    const newsItems = items.map(item => {
      const anchor = item.querySelector('.flexposts__story-title a');
      if (!anchor) return null;
      
      const title = anchor.textContent.trim();
      
      // Filter by keywords
      if (!(title.includes("dollar") ||
            title.includes("Federal Reserve") ||
            title.includes("US") ||
            title.includes("USD") ||
            title.includes("China") ||
            title.includes("interest rate") ||
            title.includes("monetary policy") ||
            title.includes("US economy") ||
            title.includes("Trump"))) {
        return null;
      }
      
      const timeSpan = item.querySelector('span.flexposts__nowrap.flexposts__time.nowrap');
      const timeText = timeSpan ? timeSpan.textContent.trim() : "";
      console.log(`Time Text: "${timeText}"`);
      
      const impactSpan = item.querySelector('span.flexposts__storyimpact'); 
      const impactClass = impactSpan ? impactSpan.className : "";
      const isHighImpact = impactClass.includes('flexposts__storyimpact--high'); 
      const isMediumImpact = impactClass.includes('flexposts__storyimpact--medium'); 
      
      // Only return the news item if it is high impact
      if (!isHighImpact && !isMediumImpact) return null;
      
      return { title, time: timeText, isHighImpact, isMediumImpact };
    }).filter(item => item !== null);
    
    return newsItems;
  });
  
  // Post the Forex Factory data (wrap in an object with an origin and a data array)
  add_temp_database({ origin: "FOREX", data: forexFactoryNews });
  
  return forexFactoryNews;
};


// ------------------------------
// Twitter Scraper Function
// ------------------------------
async function scrapeUserTweets(page, keywords) {
  const searchUrl = 'https://x.com/DBNewswire';
  await page.goto(searchUrl, { waitUntil: 'networkidle2' });
  await page.waitForSelector('article', { timeout: 30000 });

  // Compute today's date in YYYY-MM-DD format
  const today = new Date();
  const currentDate = today.toISOString().split('T')[0];

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
// Main Function: Set Up Two Pages and Start Continuous Scraping
// ------------------------------
(async () => {
   const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", // Adjust path if needed
    headless: false,
  });
  
  const forexPage = await browser.newPage();
  const twitterPage = await browser.newPage();
  
  // Set a realistic user agent and viewport for both pages
  const userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36';
  await forexPage.setUserAgent(userAgent);
  await forexPage.setViewport({ width: 1280, height: 800 });
  // await twitterPage.setUserAgent(userAgent);
  // await twitterPage.setViewport({ width: 1280, height: 800 });
  
  // Go to Twitter login page, wait for articles to load
  await twitterPage.goto('https://x.com/login', { waitUntil: 'networkidle2' });
  
  await twitterPage.waitForSelector('article', { timeout: 0 });
  
  const keywords = ["dollar", "Federal Reserve", "interest rate", "monetary policy", "US economy", "S&P", "USD", "US", "TRUMP"];
  
  // const initialForexNews = await forexFactoryScraper(forexPage);
  // console.log('Initial Forex Factory News:', initialForexNews);
  
  const initialTweets = await scrapeUserTweets(twitterPage, keywords);
  console.log("Initial Filtered Tweets:", initialTweets);
  
  // startAutoRefreshWithScrape(forexPage, "Forex Factory Page", forexFactoryScraper);
  startAutoRefreshWithScrape(twitterPage, "Twitter Page", scrapeUserTweets, keywords);
  
})();
