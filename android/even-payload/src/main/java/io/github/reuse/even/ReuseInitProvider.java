package io.github.reuse.even;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;

/** Startup entry point only; exposes no data or operations to other applications. */
public final class ReuseInitProvider extends ContentProvider {
    @Override public boolean onCreate() {
        if (getContext() != null) ReuseBridge.initialize(getContext());
        return true;
    }

    @Override public Cursor query(Uri uri, String[] projection, String selection,
            String[] selectionArgs, String sortOrder) { return null; }
    @Override public String getType(Uri uri) { return null; }
    @Override public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("Initialization provider has no insert API");
    }
    @Override public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Initialization provider has no delete API");
    }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Initialization provider has no update API");
    }
}
