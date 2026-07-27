// Minimal compatibility shim for the one Abseil macro used by upstream
// MuJoCo's python/mujoco/threadpool.{h,cc}, so that those files can be kept
// byte-identical with upstream without adding a real Abseil dependency.
//
// ABSL_CONST_INIT is empty on all supported compilers (it only expands to
// `constexpr` for an old MSVC 2017 workaround in real Abseil).
//
// The include guard matches real Abseil's <absl/base/attributes.h> on
// purpose: if real Abseil is ever on the include path, whichever copy is
// included first wins and the other is skipped harmlessly.
#ifndef ABSL_BASE_ATTRIBUTES_H_
#define ABSL_BASE_ATTRIBUTES_H_

#define ABSL_CONST_INIT

#endif  // ABSL_BASE_ATTRIBUTES_H_
