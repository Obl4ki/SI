use itertools::Itertools;
use std::fmt::Display;
use std::ops::Deref;
use std::ops::DerefMut;

pub struct Solution<T> {
    data: T,
}

fn solutions_formatter<T>(solutions: &[Vec<T>]) -> String
where
    T: std::fmt::Display,
{
    solutions
        .iter()
        .enumerate()
        .map(|(idx, solution)| {
            format!(
                "\n{}:\t{}{}{}",
                idx + 1,
                "[",
                solution.iter().join(", "),
                "]"
            )
        })
        .collect()
}

impl<T> Display for Solution<Vec<Vec<T>>>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", solutions_formatter(&self.data))
    }
}

impl<T> Deref for Solution<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Solution<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Solution<Vec<Vec<T>>> {
    pub fn new(data: Vec<Vec<T>>) -> Self {
        Self { data }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}
