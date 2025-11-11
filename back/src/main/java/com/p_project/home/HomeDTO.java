package com.p_project.home;

import com.p_project.writing.WritingSessionDTO;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class HomeDTO {

    private String name;
    private String nickName;
    private LocalDateTime writingTime;
    private Integer diaryNum;
    private Integer bookReportNum;
    private Integer totalNum;
    private List<WritingSessionDTO> writingSessionDTOS;

}
