mod plot;
mod pretty_printer;
use itertools::{chain, Itertools};
use plot::Plotter;
use pretty_printer::Solution;
use std::time::Instant;

type Queen = u8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_n_queens = 4;
    let up_to_n_queens = 14;

    let mut stats = vec![];

    for n_queens in start_n_queens..=up_to_n_queens {
        let results = get_all_pacifists(n_queens);
        println!("Solutions count: {}", results.0.len());
        stats.push(results);
    }

    let generates = stats.iter().map(|stat| stat.1 as f64).collect_vec();

    Plotter::from(generates).plot(
        "Number of children generated",
        "histogram_generated_children.svg",
        start_n_queens as f64..up_to_n_queens as f64,
    )?;

    let children = stats.iter().map(|stat| stat.2 as f64).collect_vec();

    Plotter::from(children).plot(
        "Number of checks (log10)",
        "histogram_checked_children.svg",
        start_n_queens as f64..up_to_n_queens as f64,
    )?;

    let timers = stats.iter().map(|stat| stat.3).collect_vec();

    Plotter::from(timers).plot(
        "Search time (secs, log10)",
        "timings.svg",
        start_n_queens as f64..up_to_n_queens as f64,
    )?;

    Ok(())
}

fn can_any_queen_beat(queens: &[Queen]) -> bool {
    let enumerated_queens = queens.iter().enumerate();
    for ((idx1, queen1), (idx2, queen2)) in enumerated_queens
        .clone()
        .cartesian_product(enumerated_queens)
    {
        if idx1 == idx2 {
            continue;
        }

        if queen1 == queen2 {
            return true;
        }

        if idx1.abs_diff(idx2) == queen1.abs_diff(*queen2) as usize {
            return true;
        }
    }
    false
}

fn get_all_pacifists(n_queens: usize) -> (Solution<Vec<Vec<Queen>>>, usize, usize, f64) {
    println!();
    println!("For n={}", n_queens);
    let now = Instant::now();

    // Include empty board in counter
    let mut n_generated_states = 1;
    let mut n_checked = 1;

    let mut queue: Vec<Vec<Queen>> = Vec::new();
    for i in 0..n_queens {
        queue.push(vec![i as u8]);
        n_generated_states += 1;
    }
    println!("Start state:  {:?}", queue[0]);
    let mut correct_solutions = Vec::new();

    while let Some(columns) = queue.pop() {
        let depth = columns.len();
        if depth > n_queens {
            continue;
        }

        if can_any_queen_beat(&columns) {
            continue;
        }

        let deciding_queen = columns[depth - 1];

        for new_queen in get_noncovered(deciding_queen, n_queens) {
            let mut next_columns = columns.clone();
            next_columns.push(new_queen);
            n_generated_states += 1;

            queue.push(next_columns);
        }

        n_checked += 1;
        if columns.len() == n_queens {
            correct_solutions.push(columns);
        }
    }

    println!("Number of generated states:   {}", n_generated_states);
    println!("Number of checks: {}", n_checked);
    let elapsed = now.elapsed();
    println!("Elapsed:  {:?}", elapsed);
    (
        Solution::new(correct_solutions),
        n_generated_states,
        n_checked,
        elapsed.as_secs_f64(),
    )
}

fn get_noncovered(pos: Queen, n: usize) -> impl Iterator<Item = Queen> {
    let lower_part = 0..pos.saturating_sub(1);
    let higher_part = (pos + 2)..(n as u8);
    chain!(lower_part, higher_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_noncovered() {
        let n_queens = 8;
        assert!(Iterator::eq(
            get_noncovered(4, n_queens),
            vec![0, 1, 2, 6, 7]
        ));
        assert!(Iterator::eq(
            get_noncovered(0, n_queens),
            vec![2, 3, 4, 5, 6, 7]
        ));
        assert!(Iterator::eq(
            get_noncovered(7, n_queens),
            vec![0, 1, 2, 3, 4, 5]
        ));
        assert!(Iterator::eq(
            get_noncovered(6, n_queens),
            vec![0, 1, 2, 3, 4]
        ));
    }

    #[test]
    fn solution_correctness_proof() {
        assert!(get_all_pacifists(8).0.len() == 92);

        let all_solutions = get_all_pacifists(8);
        assert!(all_solutions.0.contains(&vec![4, 2, 0, 6, 1, 7, 5, 3]));
    }
}
