use anyhow::Result;
use clap::Parser;
use petgraph::unionfind::UnionFind;
use plotters::prelude::*;
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use std::collections::HashMap;
use std::f64::consts::PI;
use uuid::Uuid;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "img_sim_clean_new.pkl")]
    sim_map: String,
    #[arg(long, default_value = "similarity.png")]
    output: String,
    #[arg(long, default_value_t = 0.8)]
    threshold: f32,
    #[arg(long, default_value_t = 1200)]
    size: u32,
    #[arg(long, value_delimiter = ',')]
    #[arg(value_parser = clap::value_parser!(Uuid))]
    ids: Vec<Uuid>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sim_explorer: PointExplorer<f32, 768> =
        PointExplorerBuilder::new().path(&args.sim_map).build()?;
    if args.ids.len() < 2 {
        eprintln!("need at least two ids");
        return Ok(());
    }

    let size = args.size;
    let center = (size as f64 / 2.0, size as f64 / 2.0);
    let radius = size as f64 * 0.4;

    let mut positions: HashMap<Uuid, (i32, i32)> = HashMap::new();
    for (i, id) in args.ids.iter().enumerate() {
        let angle = 2.0 * PI * i as f64 / args.ids.len() as f64;
        let x = (center.0 + radius * angle.cos()).round() as i32;
        let y = (center.1 + radius * angle.sin()).round() as i32;
        positions.insert(*id, (x, y));
    }

    let root = BitMapBackend::new(&args.output, (size, size)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut union_find = UnionFind::new_empty();
    // draw edges
    for i in 0..args.ids.len() {
        for j in i + 1..args.ids.len() {
            let id1 = args.ids[i];
            let id2 = args.ids[j];
            let sim = sim_explorer.get_cosine_sim((&id1, &id2))?;
            let (x1, y1) = positions[&id1];
            let (x2, y2) = positions[&id2];
            let low = sim < args.threshold;
            let color = if low { RED } else { BLUE };
            if low {
                println!("low similarity: {id1} <-> {id2}: {sim:.4}");
            } else {
                union_find.union(i, j);
            }
            root.draw(&PathElement::new(
                vec![(x1, y1), (x2, y2)],
                color.stroke_width(2),
            ))?;
        }
    }

    // draw nodes
    for id in &args.ids {
        let (x, y) = positions[id];
        root.draw(&Circle::new((x, y), 5, BLACK.filled()))?;
        root.draw(&Text::new(
            id.to_string(),
            (x + 5, y + 5),
            ("sans-serif", 15).into_font(),
        ))?;
    }

    // visualize union-find components
    let v_union_find = union_find.into_labeling();
    println!("v_union_find: {:#?}", v_union_find);

    root.present()?;
    println!("saved visualization to {}", args.output);
    Ok(())
}
