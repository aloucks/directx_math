use std::fs;
use std::env;

enum State {
    Start,
    Parameters,
    ReturnValue,
    Remarks,
    End,
}

fn main() {
    let file = fs::read_to_string(env::args().skip(1).next().expect("FILENAME")).unwrap();
    let mut lines = file.lines();

    let mut state = State::Start;

    let mut parameter_name: Option<String> = None;
    let mut parameter_desc = Vec::new();

    let mut parameters = Vec::new();

    let mut return_value = Vec::new();
    let mut remarks = Vec::new();

    loop {
        match state {
            State::Start => {
                loop {
                    match lines.next() {
                        None => break,
                        Some(line) => {
                            if "parameters" == line.trim().to_ascii_lowercase() {
                                state = State::Parameters;
                                break
                            }
                        }
                    }
                }
            }
            State::Parameters => {
                loop {
                    match lines.next() {
                        None => break,
                        Some(line) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() && trimmed.split_whitespace().count() == 1 {
                                if let Some(name) = parameter_name.clone() {
                                    parameters.push((name.to_string(), parameter_desc.clone()));
                                    parameter_desc.clear();
                                }
                                parameter_name = Some(trimmed.to_string());
                            } else if trimmed.to_ascii_lowercase() == "return value" {
                                if let Some(name) = parameter_name.clone() {
                                    parameters.push((name.to_string(), parameter_desc.clone()));
                                    parameter_desc.clear();
                                }
                                state = State::ReturnValue;
                                break;
                            } else {
                                parameter_desc.push(line.to_string());
                            }
                        }
                    }
                }
            }
            State::ReturnValue => {
                loop {
                    match lines.next() {
                        None => break,
                        Some(line) => {
                            let trimmed = line.trim();
                            if trimmed.to_ascii_lowercase() == "remarks" {
                                state = State::Remarks;
                                break;
                            } else {
                                return_value.push(line.to_string());
                            }
                        }
                    }
                }
            }
            State::Remarks => {
                loop {
                    match lines.next() {
                        None => {
                            state = State::End;
                            break;
                        }
                        Some(line) => {
                            remarks.push(line.to_string());
                        }
                    }
                }
            }
            State::End => {
                break;
            }
        }
    }

    // println!("{:?}", parameters);
    // println!("{:?}", return_value);
    // println!("{:?}", remarks);

    fn stylize_line(line: &str) -> String {
        let line = line.replace(" x ", " `x` ");
        let line = line.replace(" y ", " `y` ");
        let line = line.replace(" z ", " `z` ");
        let line = line.replace(" w ", " `w` ");

        let line = line.replace(" X ", " `X` ");
        let line = line.replace(" Y ", " `Y` ");
        let line = line.replace(" Z ", " `Z` ");
        let line = line.replace(" W ", " `W` ");

        let line = line.replace("x, ", "`x`, ");
        let line = line.replace("y, ", "`y`, ");
        let line = line.replace("z, ", "`z`, ");
        let line = line.replace("w, ", "`w`, ");

        let line = line.replace("X,", "`X`,");
        let line = line.replace("Y,", "`Y`,");
        let line = line.replace("Z,", "`Z`,");
        let line = line.replace("W,", "`W`,");

        let line = line.replace("A,", "`A`,");
        let line = line.replace("B,", "`B`,");
        let line = line.replace("C,", "`C`,");
        let line = line.replace(", D)", ", `D`)");

        let line = line.replace("Ax+By+Cz+D=0", " `Ax+By+Cz+D=0`");

        let line = line.replace(" V.", " `V`.");
        let line = line.replace(" V ", " `V` ");
        let line = line.replace(" V1 ", " `V1` ");
        let line = line.replace(" V2 ", " `V2` ");

        let line = line.replace(" M ", " `M` ");
        let line = line.replace(" M1", " `M1`");
        let line = line.replace(" M2", " `M2`");

        let line = line.replace(" Q.", " `Q`.");

        let line = line.replace(" Q ", " `Q` ");
        let line = line.replace(" Q0 ", " `Q0` ");
        let line = line.replace(" Q1 ", " `Q1` ");
        let line = line.replace(" Q2 ", " `Q2` ");

        let line = line.replace("Est f", "`Est` f");

        let line = line.replace("true", "`true`");
        let line = line.replace("false", "`false`");

        let line = line.replace(" S2", " `S2`");
        let line = line.replace(" S1", " `S1`");

        let line = line.replace("P2", " `P2`");
        let line = line.replace("P1", " `P1`");

        let line = line.replace(" P ", " `P` ");

        let line = line.replace("x-axis", "`x-axis`");
        let line = line.replace("y-axis", "`y-axis`");
        let line = line.replace("z-axis", "`z-axis`");
        let line = line.replace("w-axis", "`w-axis`");

        let line = line.replace("x-axes", "`x-axes`");
        let line = line.replace("y-axes", "`y-axes`");
        let line = line.replace("z-axes", "`z-axes`");
        let line = line.replace("w-axes", "`w-axes`");

        let line = line.replace(" NaN", " `NaN`");
        let line = line.replace(" QNaN", " `QNaN`");

        let line = line.replace(" XM_PI", " `XM_PI`");
        let line = line.replace(" -XM_PI", " `-XM_PI`");

        let line = line.replace(" 0.0f", " `0.0`");
        let line = line.replace(" 1.0f", " `1.0`");

        let line = line.replace(" x-component", " `x-component`");
        let line = line.replace(" y-component", " `y-component`");
        let line = line.replace(" z-component", " `z-component`");
        let line = line.replace(" w-component", " `w-component`");

        let line = line.replace(" 0 ", " `0` ");
        let line = line.replace(" 1 ", " `1` ");
        let line = line.replace(" 2 ", " `2` ");
        let line = line.replace(" 3 ", " `3` ");

        

        return line.trim().to_string();
    }

    fn word_wrap(s: &str) -> Vec<String> {
        let mut lines = Vec::new();
        let mut words = s.split_ascii_whitespace();
        let mut line = String::new();
        while let Some(word) = words.next() {
            line.push_str(word);
            line.push_str(" ");
            if line.len() >= 100 {
                lines.push(line.trim().to_string());
                line.clear();
            }
        }
        if !line.is_empty() {
            lines.push(line.trim().to_string());
        }
        lines
    }

    println!("///");
    println!("/// ## Parameters");
    println!("///");
    for (name, desc) in parameters.iter() {
        print!("/// `{}` ", name);
        let mut formatted_desc = String::new();
        for line in desc.iter() {
            let line = stylize_line(&line);
            formatted_desc.push_str(&format!("{}", line));
        }
        let formatted_desc = formatted_desc.trim();

        for (i, line) in word_wrap(&formatted_desc).iter().enumerate() {
            if i > 0 {
                print!("/// ");
            }
            println!("{}", line);
        }
        println!("///")
    }

    println!("/// ## Return value");
    println!("///");
    for line in return_value.iter() {
        // println!("/// {}", stylize_line(&line));
        for line in word_wrap(line) {
            println!("/// {}", stylize_line(&line));
        }
    }

    println!("///");
    if !remarks.is_empty() {
        println!("/// ## Remarks");
        println!("///");
        for line in remarks.iter() {
            // println!("/// {}", stylize_line(&line));
            for line in word_wrap(line) {
                println!("/// {}", stylize_line(&line));
            }
            if line.is_empty() {
                println!("///");
            }
        }
        println!("///");
    }
    println!("/// ## Reference");
    println!("///");
}