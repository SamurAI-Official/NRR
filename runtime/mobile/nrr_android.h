/**
 * @file nrr_android.h
 * @brief Android NDK Platform Integration
 *
 * JNI bridge for native Android app integration with NRR.
 */

#ifndef NRR_ANDROID_H
#define NRR_ANDROID_H

#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Android-specific device flags */
#define NRR_ANDROID_DEVICE_DEFAULT  0
#define NRR_ANDROID_DEVICE_ADRENO   1
#define NRR_ANDROID_DEVICE_MALI     2
#define NRR_ANDROID_DEVICE_POWERVR  3

/* Lifecycle management */
JNIEXPORT jint JNICALL Java_com_samurai_nrr_NRRDevice_nativeCreate(
    JNIEnv* env, jobject thiz, jobject surface, jint device_type);

 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeDestroy(
    JNIEnv* env, jobject thiz, jint handle);

/* Rendering */
 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnSurfaceCreated(
    JNIEnv* env, jobject thiz, jint handle);

 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnSurfaceChanged(
    JNIEnv* env, jobject thiz, jint handle, jint width, jint height);

 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnDrawFrame(
    JNIEnv* env, jobject thiz, jint handle);

/* Touch input */
 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnTouchEvent(
    JNIEnv* env, jobject thiz, jint handle, jint action, jfloat x, jfloat y);

/* Lifecycle callbacks from Android Activity */
 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnPause(
    JNIEnv* env, jobject thiz, jint handle);

 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeOnResume(
    JNIEnv* env, jobject thiz, jint handle);

/* Thermal throttling management */
 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeSetThermalStatus(
    JNIEnv* env, jobject thiz, jint handle, jint thermal_level);

/* Memory management */
 JNIEXPORT void JNICALL Java_com_samurai_nrr_NRRDevice_nativeTrimMemory(
    JNIEnv* env, jobject thiz, jint handle, jint trim_level);

#ifdef __cplusplus
}
#endif

#endif /* NRR_ANDROID_H */