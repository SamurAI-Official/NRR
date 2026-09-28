/**
 * @file nrr_test_backend.h
 * @brief The NRR_TEST_BACKEND override - test-only, never a product option
 *
 * "Works for both execution paths" needs both paths to be *runnable in one environment*, and by
 * default they are not: automatic selection picks a vendor accelerator where one exists and the
 * CPU backend otherwise, so a CUDA host runs the accelerator path for the whole suite and CI runs
 * the CPU path for the whole suite. NRR_TEST_BACKEND makes that choice explicit for a run:
 *
 *   NRR_TEST_BACKEND=auto     the default; nothing changes
 *   NRR_TEST_BACKEND=cpu      automatic selection resolves to the CPU backend, so the whole suite
 *                             exercises BackendCPU::execute_model on a machine that would
 *                             otherwise select an accelerator
 *   NRR_TEST_BACKEND=kernel   a CPU device executes its frames through
 *                             AcceleratorExecutionKernel::execute_frame, with the CPU backend's
 *                             host-memory textures as the kernel's resources, so the whole suite
 *                             exercises the accelerator execution path on a machine with no
 *                             accelerator at all - which is every CI runner
 *   NRR_TEST_BACKEND=<name>   any registered backend name (nvidia, amd, intel, riscv, ...), as if
 *                             the caller had set NRRDeviceOptions::preferred_backend
 *
 * This is not a product feature and it claims no capability: it selects which code executes, and
 * what the hardware can do is still whatever the backend measured. It is read on every frame
 * rather than cached, so one process can run both paths and a test can switch between them; that
 * is why it is documented in docs/roadmap.md rather than in the public header.
 */

#ifndef NRR_TEST_BACKEND_H
#define NRR_TEST_BACKEND_H

namespace nrr {

/* The raw value of NRR_TEST_BACKEND, or "" when it is unset. Never null. */
const char* test_backend_override();

/* True when the override is "kernel": a CPU device must execute frames through the shared
 * accelerator kernel. */
bool test_route_through_accel_kernel();

} // namespace nrr

#endif /* NRR_TEST_BACKEND_H */
