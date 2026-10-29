// 1. Load environment variables FIRST
const path = require('path');
// Explicitly pointing to the root directory file to guarantee it loads
require('dotenv').config({ path: path.join(__dirname, '../.env') }); 

// Debugging check to catch issues before Passport crashes
console.log("🛠️ DEBUG - GOOGLE_CLIENT_ID:", process.env.GOOGLE_CLIENT_ID ? "LOADED SUCCESS" : "UNDEFINED");

const express = require('express');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
// cookie-session doesn't implement req.session.regenerate/save, which
// Passport >=0.6 requires — that combo throws "req.session.regenerate is
// not a function" the moment a user logs in. express-session implements
// the full session-store interface Passport expects.
const session = require('express-session');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors({
  origin: 'http://localhost:5173', // Vite dev server
  credentials: true
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session configuration
app.use(session({
  name: 'session',
  secret: process.env.SESSION_SECRET || 'your-secret-key',
  resave: false,
  saveUninitialized: false,
  cookie: {
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    httpOnly: true,
    sameSite: 'lax',
  },
}));

// Passport middleware
app.use(passport.initialize());
app.use(passport.session());

// 2. Google OAuth Strategy (SINGLE INITIALIZATION ONLY)
passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,         
    clientSecret: process.env.GOOGLE_CLIENT_SECRET, 
    callbackURL: "http://localhost:5000/auth/google/callback" // Note: Match your backend PORT 5000!
  },
  function(accessToken, refreshToken, profile, done) {
    console.log('Google profile loaded:', profile.id);
    return done(null, profile);
  }
));

// Serialize user for session management
passport.serializeUser((user, done) => {
  done(null, user);
});

passport.deserializeUser((user, done) => {
  done(null, user);
});

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Vienkō Backend API is running!' });
});

// Google OAuth routes
app.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'] })
);

app.get('/auth/google/callback',
  passport.authenticate('google', { failureRedirect: 'http://localhost:5173/login' }),
  (req, res) => {
    res.redirect('http://localhost:5173/');
  }
);

// User profile route
app.get('/api/user', (req, res) => {
  if (req.user) {
    res.json({
      success: true,
      user: {
        id: req.user.id,
        name: req.user.displayName,
        email: req.user.emails[0].value,
        picture: req.user.photos[0].value
      }
    });
  } else {
    res.json({ success: false, message: 'Not authenticated' });
  }
});

// Logout route
app.get('/auth/logout', (req, res) => {
  req.logout((err) => {
    if (err) {
      return res.status(500).json({ success: false, message: 'Logout failed' });
    }
    res.redirect('http://localhost:5173/');
  });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📱 Frontend should be running on http://localhost:5173`);
});
