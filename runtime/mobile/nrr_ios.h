/**
 * @file nrr_ios.h
 * @brief iOS Platform Integration
 *
 * Objective-C++ bridge for native iOS app integration with NRR.
 */

#ifndef NRR_IOS_H
#define NRR_IOS_H

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

/* iOS-specific device flags */
#define NRR_IOS_DEVICE_DEFAULT      0
#define NRR_IOS_DEVICE_A12_BIONIC  11  /* A12 and later have Neural Engine */
#define NRR_IOS_DEVICE_A13_BIONIC  12
#define NRR_IOS_DEVICE_A14_BIONIC  13
#define NRR_IOS_DEVICE_A15_BIONIC  14
#define NRR_IOS_DEVICE_A16_BIONIC  15
#define NRR_IOS_DEVICE_A17_BIONIC  16
#define NRR_IOS_DEVICE_M1          20
#define NRR_IOS_DEVICE_M2          21
#define NRR_IOS_DEVICE_M3          22
#define NRR_IOS_DEVICE_M4          23

/* Neural Engine availability */
typedef NS_ENUM(NSInteger, NRRNeuralEngineState) {
    NRRNeuralEngineStateUnavailable = 0,
    NRRNeuralEngineStateAvailable,
    NRRNeuralEngineStateThrottled
};

/* NRR Device interface for iOS */
@interface NRRDevice : NSObject

@property (nonatomic, readonly) NSInteger deviceHandle;
@property (nonatomic, readonly) NRRNeuralEngineState neuralEngineState;

- (instancetype)initWithDeviceID:(NSInteger)deviceID;
- (BOOL)isSupported;
- (void)destroy;

@end

/* NRR Renderer for iOS Metal integration */
@interface NRRRenderer : NSObject

@property (nonatomic, strong, readonly) id<MTLDevice> metalDevice;
@property (nonatomic, weak) MTKView* view;

- (instancetype)initWithDevice:(NRRDevice*)device view:(MTKView*)view;
- (void)drawInView:(MTKView*)view;
- (void)drawableSizeWillChange:(CGSize)size;
- (void)pause;
- (void)resume;

@end

#endif /* NRR_IOS_H */