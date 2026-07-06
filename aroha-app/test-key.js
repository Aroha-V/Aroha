const fs = require('fs');
const path = require('path');

// Manually parse .env.local to avoid requiring the dotenv package
let rawApiKey = '';
let projectId = '';

try {
  const envContent = fs.readFileSync(path.resolve(__dirname, '.env.local'), 'utf8');
  const lines = envContent.split('\n');
  for (const line of lines) {
    if (line.trim().startsWith('NEXT_PUBLIC_FIREBASE_API_KEY=')) {
      rawApiKey = line.split('=')[1].trim();
    }
    if (line.trim().startsWith('NEXT_PUBLIC_FIREBASE_PROJECT_ID=')) {
      projectId = line.split('=')[1].trim();
    }
  }
} catch (e) {
  console.error('❌ Error reading .env.local file:', e.message);
  process.exit(1);
}

// Clean quotes
const apiKey = rawApiKey ? rawApiKey.replace(/['"]/g, '') : null;
projectId = projectId ? projectId.replace(/['"]/g, '') : null;

console.log('====================================');
console.log('🔍 Firebase Diagnostic Tool');
console.log('====================================');
console.log('Loaded from .env.local:');
console.log(`- Raw API Key:      ${rawApiKey ? `'${rawApiKey}'` : 'UNDEFINED'}`);
console.log(`- Cleaned API Key:  ${apiKey ? `'${apiKey}'` : 'UNDEFINED'}`);
console.log(`- Project ID:       ${projectId ? `'${projectId}'` : 'UNDEFINED'}`);
console.log('------------------------------------');

if (!apiKey) {
  console.error('❌ Error: NEXT_PUBLIC_FIREBASE_API_KEY is not defined in .env.local');
  process.exit(1);
}

const url = `https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${apiKey}`;

console.log('📡 Testing connection to Google Identity Toolkit...');
fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({}),
})
  .then(async (res) => {
    const data = await res.json();
    console.log(`\nResponse Status: ${res.status}`);
    
    if (res.status === 400 && data.error?.message?.includes('API key not valid')) {
      console.error('\n❌ RESULT: Google rejects this API Key as INVALID.');
      console.error('Check if there is a typo, if the key was created under a different project,');
      console.error('or if the key is restricted in your Google Cloud Console.');
    } else if (data.error?.message?.includes('MISSING_EMAIL') || data.error?.message?.includes('OPERATION_NOT_ALLOWED')) {
      console.log('\n✅ RESULT: Google accepts this API Key!');
      console.log('This means the key itself is 100% valid.');
      console.log('If you still see the error in the browser, please clear your browser cache,');
      console.log('use an Incognito window, or restart Next.js.');
    } else {
      console.log('\nℹ️ Response from Google:', JSON.stringify(data, null, 2));
    }
  })
  .catch((err) => {
    console.error('❌ Network Error:', err.message);
  });
