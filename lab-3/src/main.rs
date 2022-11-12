mod core_types;
mod heuristics;
mod informed_search;
mod plot;

use crate::heuristics::Heuristic;
use crate::informed_search::solve_with_heuristic;
use itertools::Itertools;
use plot::Plotter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_n_queens = 4;
    let up_to_n_queens = 13;

    let titles_and_heuristics: Vec<(&str, Box<dyn Heuristic>)> = vec![
        ("middle-first", Box::new(heuristics::H1)),
        ("infeasible", Box::new(heuristics::H2)),
        ("hamming", Box::new(heuristics::H3)),
        ("manhattan", Box::new(heuristics::H4)),
    ];

    for (title, heuristic) in titles_and_heuristics {
        let mut stats = vec![];
        println!("For heuristic {title}: ");
        for n_queens in start_n_queens..=up_to_n_queens {
            let results = solve_with_heuristic(n_queens, heuristic.as_ref());
            println!("Solutions count: {}", results.solutions.len());
            println!(
                "Steps to first solution: {}",
                results.checks_to_find_one_solution
            );
            stats.push(results);
        }

        let generates = stats
            .iter()
            .map(|stat| stat.generated_children as f64)
            .collect_vec();

        Plotter::from(generates).plot(
            "Number of children generated",
            &format!("plots/{title}_histogram_generated_children.svg"),
            start_n_queens as f64..up_to_n_queens as f64,
        )?;

        let checks = stats.iter().map(|stat| stat.checks as f64).collect_vec();

        Plotter::from(checks).plot(
            "Number of checks (log10)",
            &format!("plots/{title}_histogram_checked_children.svg"),
            start_n_queens as f64..up_to_n_queens as f64,
        )?;

        let timers = stats.iter().map(|stat| stat.seconds).collect_vec();

        Plotter::from(timers).plot(
            "Search time (secs, log10)",
            &format!("plots/{title}_timings.svg"),
            start_n_queens as f64..up_to_n_queens as f64,
        )?;

        let steps_to_first = stats
            .iter()
            .map(|stat| stat.checks_to_find_one_solution as f64)
            .collect_vec();

        Plotter::from(steps_to_first).plot(
            "Steps to first correct solution (log10)",
            &format!("plots/{title}_steps_to_find_first_solution.svg"),
            start_n_queens as f64..up_to_n_queens as f64,
        )?;
    }

    Ok(())
}
