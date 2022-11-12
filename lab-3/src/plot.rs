use itertools::Itertools;
use plotters::prelude::*;
use std::ops::Range;

pub(crate) struct Plotter<T> {
    pub(crate) data: Vec<T>,
}

impl From<Vec<f64>> for Plotter<f64> {
    fn from(data: Vec<f64>) -> Self {
        Self { data }
    }
}

impl Plotter<f64> {
    pub fn plot(
        &mut self,
        title: &str,
        file_name: &str,
        n_domain: Range<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(&file_name, (1366, 768)).into_drawing_area();
        root.fill(&WHITE)?;
        let data = self
            .data
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .collect_vec();

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(50)
            .y_label_area_size(50)
            .margin(30)
            .caption(title, ("sans-serif", 50.0))
            .build_cartesian_2d(
                n_domain.start..n_domain.end + 1.,
                (0.0..*self.data.last().unwrap()).log_scale(),
            )?;

        chart
            .configure_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .axis_desc_style(("sans-serif", 28))
            .y_label_formatter(&|x| format!("{:e}", x))
            .x_label_formatter(&|x| format!("{:.0}", x))
            .draw()?;

        let d = (n_domain.start as usize..)
            .zip(&data)
            .map(|(idx, x)| (idx as f64, **x));

        let lines = LineSeries::new(d.clone(), RED).point_size(5);

        chart.draw_series(lines).unwrap();

        chart.draw_series(PointSeries::of_element(d, 5, RED, &|coord, size, style| {
            return EmptyElement::at(coord)
                + Circle::new((0, 0), size, style.filled())
                + Text::new(
                    format!("{:?}", coord),
                    (10, 0),
                    ("sans-serif", 10).into_font(),
                );
        }))?;
        root.present()?;
        Ok(())
    }
}
