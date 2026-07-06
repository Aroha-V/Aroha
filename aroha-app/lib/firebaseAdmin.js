import admin from 'firebase-admin';

function initAdmin() {
  if (admin.apps.length) return;

  try {
    const projectId = process.env.FIREBASE_PROJECT_ID?.replace(/['"]/g, '');
    const clientEmail = process.env.FIREBASE_CLIENT_EMAIL?.replace(/['"]/g, '');
    let privateKey = process.env.FIREBASE_PRIVATE_KEY;

    if (privateKey) {
      console.log('🔑 [Firebase admin debug] Raw key stats:', {
        length: privateKey.length,
        startsWithQuote: privateKey.startsWith('"') || privateKey.startsWith("'"),
        endsWithQuote: privateKey.endsWith('"') || privateKey.endsWith("'"),
        hasEscapedNewlines: privateKey.includes('\\n'),
        hasRealNewlines: privateKey.includes('\n'),
      });

      // ── Clean the PEM Private Key aggressively ──
      // 1. Strip any enclosing double or single quotes
      privateKey = privateKey.replace(/^['"]|['"]$/g, '');
      // 2. Normalize escaped newlines
      privateKey = privateKey.replace(/\\n/g, '\n');
      
      // 3. Extract the Base64 body, strip ALL whitespace and accidental backslashes from it, and reconstruct the PEM key
      const pemMatch = privateKey.match(/-----BEGIN PRIVATE KEY-----([\s\S]+?)-----END PRIVATE KEY-----/);
      if (pemMatch) {
        const cleanBase64 = pemMatch[1].replace(/\s/g, '').replace(/\\/g, ''); // Strips all whitespace and backslashes
        privateKey = `-----BEGIN PRIVATE KEY-----\n${cleanBase64}\n-----END PRIVATE KEY-----`;
      } else {
        // Fallback: just strip whitespace if headers are not matched
        privateKey = privateKey.replace(/\s/g, '').replace(/\\/g, '');
      }
      
      console.log('🔑 [Firebase admin debug] Cleaned key stats:', {
        length: privateKey.length,
        hasEscapedNewlines: privateKey.includes('\\n'),
        hasRealNewlines: privateKey.includes('\n'),
        hasPlusSigns: privateKey.includes('+'),
        hasSpaces: privateKey.includes(' '),
        preview: privateKey ? `${privateKey.substring(0, 110)}...[TRUNCATED]...${privateKey.substring(privateKey.length - 30)}` : 'none',
      });
    }

    if (!projectId || !clientEmail || !privateKey) {
      throw new Error(`Missing credentials. projectId: ${!!projectId}, clientEmail: ${!!clientEmail}, privateKey: ${!!privateKey}`);
    }

    admin.initializeApp({
      credential: admin.credential.cert({
        projectId,
        clientEmail,
        privateKey,
      }),
    });
    console.log('✅ Firebase Admin SDK initialized successfully!');
  } catch (err) {
    console.error('❌ Failed to initialize Firebase Admin SDK:', err.message);
  }
}

/**
 * Verify a Firebase ID token and return the decoded payload.
 * Returns null if the token is invalid or expired.
 */
export async function verifyFirebaseToken(idToken) {
  try {
    initAdmin();
    if (!admin.apps.length) {
      console.error('❌ Firebase Admin SDK not initialized.');
      return null;
    }
    return await admin.auth().verifyIdToken(idToken);
  } catch (err) {
    console.error('❌ Firebase Token verification failed:', err.message);
    return null;
  }
}
