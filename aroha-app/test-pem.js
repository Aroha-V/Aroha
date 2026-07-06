const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Read the raw private key from .env.local
let rawPrivateKey = '';
try {
  const envContent = fs.readFileSync(path.resolve(__dirname, '.env.local'), 'utf8');
  const lines = envContent.split('\n');
  for (const line of lines) {
    if (line.trim().startsWith('FIREBASE_PRIVATE_KEY=')) {
      rawPrivateKey = line.split('FIREBASE_PRIVATE_KEY=')[1].trim();
    }
  }
} catch (e) {
  console.error('❌ Error reading .env.local file:', e.message);
  process.exit(1);
}

if (!rawPrivateKey) {
  console.error('❌ Error: FIREBASE_PRIVATE_KEY is not defined in .env.local');
  process.exit(1);
}

// Clean it exactly like firebaseAdmin.js does
let privateKey = rawPrivateKey.replace(/^['"]|['"]$/g, '');
privateKey = privateKey.replace(/\\n/g, '\n');

const pemMatch = privateKey.match(/-----BEGIN PRIVATE KEY-----([\s\S]+?)-----END PRIVATE KEY-----/);
if (pemMatch) {
  const cleanBase64 = pemMatch[1].replace(/\s/g, '').replace(/\\/g, ''); // Strips all whitespace and backslashes
  privateKey = `-----BEGIN PRIVATE KEY-----\n${cleanBase64}\n-----END PRIVATE KEY-----`;
} else {
  privateKey = privateKey.replace(/\s/g, '').replace(/\\/g, '');
}

console.log('====================================');
console.log('🔍 PEM Private Key Test');
console.log('====================================');
console.log('PEM Key Length:', privateKey.length);
console.log('PEM Key First 150 chars:\n', privateKey.substring(0, 150));
console.log('------------------------------------');

try {
  crypto.createPrivateKey(privateKey);
  console.log('✅ RESULT: Node.js CRYPTO SUCCESSFULLY PARSED YOUR PRIVATE KEY!');
  console.log('This means the key formatting and base64 structure are 100% valid PEM!');
} catch (err) {
  console.error('❌ RESULT: Node.js crypto FAILED to parse the key.');
  console.error('Error details:', err.message);
  
  // Try decoding base64 to check for corrupted bytes
  if (pemMatch) {
    const cleanBase64 = pemMatch[1].replace(/\s/g, '');
    console.log('\n🔍 Analyzing Base64 Body Characters:');
    console.log('- Cleaned Base64 Body length:', cleanBase64.length);
    
    // Find any invalid base64 characters
    const invalidChars = [];
    for (let i = 0; i < cleanBase64.length; i++) {
      const char = cleanBase64[i];
      if (!/^[A-Za-z0-9+/=]$/.test(char)) {
        invalidChars.push({ char, index: i, code: char.charCodeAt(0) });
      }
    }
    
    if (invalidChars.length > 0) {
      console.error(`- ❌ FOUND ${invalidChars.length} INVALID CHARACTERS INSIDE THE BASE64 BODY:`);
      invalidChars.forEach(({ char, index, code }) => {
        console.error(`  Index ${index}: '${char}' (char code: ${code})`);
        
        // Print 30 characters of context around this index
        const start = Math.max(0, index - 20);
        const end = Math.min(cleanBase64.length, index + 20);
        console.log(`  Context around index ${index}:`);
        console.log(`  [${start}..${end}]: "${cleanBase64.substring(start, index)} >> ${char} << ${cleanBase64.substring(index + 1, end)}"`);
      });
    } else {
      console.log('- ✅ No invalid characters found in Base64 body.');
      console.log('- First 20 characters of body:', cleanBase64.substring(0, 20));
      console.log('- Last 20 characters of body:', cleanBase64.substring(cleanBase64.length - 20));
    }

    if (cleanBase64.length % 4 !== 0) {
      console.log(`- ⚠️ WARNING: Base64 length is ${cleanBase64.length}, which is not a multiple of 4!`);
    }
  }
}
