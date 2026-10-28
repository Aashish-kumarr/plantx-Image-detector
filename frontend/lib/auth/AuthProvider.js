"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  updateProfile,
  onAuthStateChanged,
} from "firebase/auth";

import { auth, isConfigured } from "./firebase";
import { setTokenProvider } from "../api/client";

export const AuthContext = createContext({
  user: null,
  loading: true,
  isConfigured: false,
  signup: async () => {},
  login: async () => {},
  logout: async () => {},
  resetPassword: async () => {},
  getIdToken: async () => null,
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState({ uid: "test_user_123", email: "test@test.com" });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Register token provider with API client
    setTokenProvider(async () => {
      if (auth && auth.currentUser) {
        return auth.currentUser.getIdToken();
      }
      return null;
    });

    if (!isConfigured || !auth) {
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);


  const signup = async (email, password, displayName = "") => {
    if (!isConfigured || !auth) {
      throw new Error(
        "Live Firebase authentication is not configured. Please set NEXT_PUBLIC_FIREBASE_* environment variables."
      );
    }
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    if (displayName && userCredential.user) {
      await updateProfile(userCredential.user, { displayName });
    }
    return userCredential.user;
  };

  const login = async (email, password) => {
    if (!isConfigured || !auth) {
      throw new Error(
        "Live Firebase authentication is not configured. Please set NEXT_PUBLIC_FIREBASE_* environment variables."
      );
    }
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return userCredential.user;
  };

  const logout = async () => {
    if (auth) {
      await signOut(auth);
    }
    setUser(null);
  };

  const resetPassword = async (email) => {
    if (!isConfigured || !auth) {
      throw new Error(
        "Live Firebase authentication is not configured. Please set NEXT_PUBLIC_FIREBASE_* environment variables."
      );
    }
    await sendPasswordResetEmail(auth, email);
  };

  const getIdToken = async () => {
    if (auth && auth.currentUser) {
      return auth.currentUser.getIdToken();
    }
    return null;
  };

  const value = {
    user,
    loading,
    isConfigured,
    signup,
    login,
    logout,
    resetPassword,
    getIdToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthContext() {
  return useContext(AuthContext);
}
