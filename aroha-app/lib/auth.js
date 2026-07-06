import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

const JWT_SECRET = process.env.JWT_SECRET;
const COOKIE_NAME = 'aroha_token';
const COOKIE_MAX_AGE = 7 * 24 * 60 * 60; // 7 days in seconds

/** Sign a JWT and return the token string */
export function signToken(payload) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: '7d' });
}

/** Verify a JWT string — returns payload or null */
export function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch {
    return null;
  }
}

/** Set the httpOnly auth cookie on a Response */
export function setAuthCookie(response, token) {
  response.headers.set(
    'Set-Cookie',
    `${COOKIE_NAME}=${token}; HttpOnly; Path=/; Max-Age=${COOKIE_MAX_AGE}; SameSite=Lax`
  );
}

/** Clear the auth cookie */
export function clearAuthCookie(response) {
  response.headers.set(
    'Set-Cookie',
    `${COOKIE_NAME}=; HttpOnly; Path=/; Max-Age=0; SameSite=Lax`
  );
}

/** Read and verify the token from the request cookies */
export function getTokenPayload(request) {
  const cookieHeader = request.headers.get('cookie') || '';
  const match = cookieHeader.match(new RegExp(`${COOKIE_NAME}=([^;]+)`));
  if (!match) return null;
  return verifyToken(match[1]);
}
