//! Compatibility shims for rayon traits, depending on whether or not `std` is enabled.

#[cfg(feature = "std")]
mod helper {
    pub use rayon::prelude::*;
}

#[cfg(not(feature = "std"))]
mod helper {
    pub trait IntoParallelIterator {
        fn into_par_iter(self) -> Self;
    }

    impl<I: IntoIterator> IntoParallelIterator for I {
        fn into_par_iter(self) -> I::IntoIter {
            self.into_iter()
        }
    }
}

pub use helper::*;

/// Macro for running a function in parallel.
#[macro_export(local_inner_macros)]
macro_rules! run_par {
    (
        $func:expr
    ) => {{
        #[cfg(feature = "std")]
        use rayon::prelude::*;

        #[cfg(feature = "std")]
        #[allow(clippy::redundant_closure_call)]
        let output = rayon::scope(|_| $func());

        #[cfg(not(feature = "std"))]
        let output = $func();

        output
    }};
}

/// Macro for iterating in parallel.
#[macro_export(local_inner_macros)]
macro_rules! iter_par {
    (
        $iter:expr
    ) => {{
        #[cfg(feature = "std")]
        use ::rayon::prelude::*;

        #[cfg(feature = "std")]
        let output = $iter.into_par_iter();

        #[cfg(not(feature = "std"))]
        let output = $iter;

        output
    }};
}

/// Macro for iterating over a range in parallel.
#[macro_export(local_inner_macros)]
macro_rules! iter_range_par {
    (
        $start:expr, $end:expr
    ) => {{
        #[cfg(feature = "std")]
        use ::rayon::prelude::*;

        #[cfg(feature = "std")]
        let output = ($start..$end).into_par_iter();

        #[cfg(not(feature = "std"))]
        let output = ($start..$end);

        output
    }};
}
